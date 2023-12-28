# Adapted from https://github.com/MolecularAI/Reinvent
import io
import subprocess
import warnings
import numpy as np
import pandas as pd
from typing import List
from loguru import logger
from tqdm import tqdm

from .permeability import Permeability
from .kras_ic50 import KRASInhibition
from utils.helm_utils import get_cycpep_smi_from_helm


class DockScorer:
    def __init__(self, conf_path='./agent/docker/config.yaml', docker_path='agent.docker.dock_adcp'):
        self.conf_path = conf_path
        self.docker_path = docker_path
        self.params = {'low': -12, 'high': -8, 'k': 0.25}

    def _check_if_all_nat_aa_and_end2end_cyclic(self, helm):
        helm_parts = helm.split('$')
        if len(helm_parts) != 5 or '{' not in helm_parts[0] or '}' not in helm_parts[0]:
            return None
        pep_idx = helm_parts[0].index('{')
        linear_helm = helm_parts[0][pep_idx+1:-1]
        # logger.info(linear_helm)

        # check if each element in linear helm is natural amino acid
        is_all_nat_aa = True
        aas = linear_helm.split('.')
        for aa in aas:
            if aa not in ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                          'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']:
                is_all_nat_aa = False
                break
        if not is_all_nat_aa:
            return None
        aa_seq = ''.join(aas)

        # check if linker is end2end cyclic
        linker = helm_parts[1].split(',')
        cyclic_linker = linker[2] if len(linker) == 3 else None
        if cyclic_linker is None:
            return aa_seq
        # logger.info(cyclic_linker)

        # check if end2end cyclic
        linker_ends = cyclic_linker.split('-')
        linker_start = linker_ends[0].split(':')[0]
        linker_end = linker_ends[1].split(':')[0]

        if linker_start == '1' and linker_end == str(len(aas)):
            # logger.info('end2end cyclic')
            aa_seq = f"cyclo({aa_seq})"
        return aa_seq
    
    def _get_aa_seq_from_helms(self, helm_seqs: list):
        """ Get aa_seq of end2end cyclic helms, with all natural amino acids"""
        valid_idxes = []
        valid_seqs = []

        for idx, helm in enumerate(helm_seqs):
            # logger.debug(f"helm: {helm}")
            aa_seq = self._check_if_all_nat_aa_and_end2end_cyclic(helm)
            if aa_seq is None:
                continue
            # logger.debug(f"aa_seq: {aa_seq}")

            # Ignore helm which cannot converted into molecules
            try:
                # check if helms valid
                smi = get_cycpep_smi_from_helm(helm)
                if smi:
                    valid_idxes.append(idx)
                    valid_seqs.append(aa_seq)
            except Exception as e:
                # logger.debug(f'Error: {e} in helm {helm}')
                pass
        
        return valid_seqs, valid_idxes

    def __call__(self, helms: List, batch_size=64):
        # logger.info(f"Num of helms: {len(helms)}")

        scores = np.zeros(len(helms))
        scores_tfd = np.zeros(len(helms)) 

        # Get valid end-to-end cyclic helms, with all natural amino acids
        valid_seqs, valid_indexes = self._get_aa_seq_from_helms(helms)
        # logger.info(f"valid_seqs: {valid_seqs}, valid_indexes: {valid_indexes}")
        # logger.debug(f"scores: {scores[np.array(valid_indexes)]}")
        if len(valid_seqs) > 0:
            scores[np.array(valid_indexes)] = self._call_adcp_dock(valid_seqs, batch_size=batch_size)
            scores_tfd[np.array(valid_indexes)] = self.transform_score(scores[np.array(valid_indexes)])

        return scores_tfd, scores
    
    def _call_adcp_dock(self, aa_seqs: List, batch_size=64):
        """ Call adcp docker to dock peptides """
        num_seqs = len(aa_seqs)
        if num_seqs <= batch_size:
            command = self.create_command(aa_seqs)
            # logger.debug(f"Command: {command}")
            result = self.submit_command(command)
        else:  # Split into batches in case too many seqs
            number_batches = (num_seqs + batch_size - 1) // batch_size
            remaining_samples = num_seqs
            batch_start = 0
            result = []
            logger.info(f"List too long, Split into {number_batches} batches of size {batch_size}")
            for i in tqdm(range(number_batches), desc='Docking'):
                batch_size = min(batch_size, remaining_samples)
                batch_end = batch_start + batch_size

                command = self.create_command(aa_seqs[batch_start:batch_end])
                result += self.submit_command(command)

                batch_start += batch_size
                remaining_samples -= batch_size

        scores = []
        for score in result:
            try:
                score = float(score)
            except ValueError:
                score = 0  # replace NA with 0
            scores.append(score)
        return np.array(scores)

    def create_command(self, aa_seqs: List):  # Initiate a docking process, which support multiprocessing
        concat_aa_seqs = '"' + ';'.join(aa_seqs) + '"'
        command = ' '.join(["python -m", self.docker_path, "--conf", self.conf_path, "--seqs", concat_aa_seqs])
        return command

    @staticmethod
    def submit_command(command):
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE, shell=True) as proc:
            wrapt_proc_in = io.TextIOWrapper(proc.stdin, 'utf-8')
            wrapt_proc_out = io.TextIOWrapper(proc.stdout, 'utf-8')
            result = [line.strip() for line in wrapt_proc_out.readlines()]
            wrapt_proc_in.close()
            wrapt_proc_out.close()
            proc.wait()
            proc.terminate()
        return result

    def transform_score(self, scores):
        def reverse_sigmoid(value, low, high, k) -> float:
            try:
                return 1 / (1 + 10 ** (k * (value - (high + low) / 2) * 10 / (high - low)))
            except:
                return 0

        scores_tf = [reverse_sigmoid(s_val, self.params['low'], self.params['high'], self.params['k']) for s_val in scores]
        return scores_tf


class ScoringFunctions:
    def __init__(self, scoring_func_names=None, score_type='weight', weights=None):
        """
            scoring_func_names: List of scoring function names, default=['HER2']
            weights: List of int weights for each scoring function, default=[1]
        """
        self.scoring_func_names = ['permeability'] if scoring_func_names is None else scoring_func_names
        self.score_type = score_type
        self.weights = np.array([1] * len(self.scoring_func_names) if weights is None else weights)
        self.all_funcs = {'permeability': Permeability, 'kras_ic50': KRASInhibition, 'beta_catenin': DockScorer}

    def scores(self, helm_seqs: List, step: int):
        scores, raw_scores = [], []
        for fn_name in self.scoring_func_names:
            # logger.debug(f"Scoring function: {fn_name}")
            score, raw_score = self.all_funcs[fn_name]()(helm_seqs)

            scores.append(score)
            raw_scores.append(raw_score)
        scores = np.float32(scores).T
        raw_scores = np.float32(raw_scores).T
        # logger.debug(f"Scores: {scores}, raw scores: {raw_scores}")

        if self.score_type == 'sum':
            final_scores = scores.sum(axis=1)
        elif self.score_type == 'product':
            final_scores = scores.prod(axis=1)
        elif self.score_type == 'weight':
            final_scores = (scores * self.weights / self.weights.sum()).sum(axis=1)
        else:
            raise Exception('Score type error!')

        np_step = np.ones(len(helm_seqs)) * step
        # logger.debug(f"Final scores: {final_scores}")
        scores_df = pd.DataFrame({'step': np_step, 'helm_seqs': helm_seqs, self.score_type: final_scores})
        scores_df[self.scoring_func_names] = pd.DataFrame(scores, index=scores_df.index)
        raw_names = [f'raw_{name}' for name in self.scoring_func_names]
        scores_df[raw_names] = pd.DataFrame(raw_scores, index=scores_df.index)
        return scores_df


def create_helm_from_aa_seq(aa_seq, cyclic=False):
    linear_helm = ".".join(aa_seq)
    helm = f"PEPTIDE1{{{linear_helm}}}$PEPTIDE1,PEPTIDE1,1:R1-{len(aa_seq)}:R2$$$" if cyclic else f"PEPTIDE1{{{linear_helm}}}$$$$"
    return helm


def unit_tests():

    # helm_seqs = ['PEPTIDE2{[Abu].[Sar].[meL].V.[meL].A.[dA].[meL].[meL].[meV].[Me_Bmt(E)]}$PEPTIDE2,PEPTIDE2,1:R1-11:R2$$$',
    #              'PEPTIDE1{[dL].[dL].L.[dL].P.Y}$PEPTIDE1,PEPTIDE1,1:R1-6:R2$$$',
    #              'PEPTIDE1{[dL].[dL].[dL].[dL].P.Y}$PEPTIDE1,PEPTIDE1,1:R1-6:R2$$$',
    #              'PEPTIDE1{L.L.L.[dL].P.Y}$PEPTIDE1,PEPTIDE1,1:R1-6:R2$$$',
    #              'PEPTIDE1{L.[dL].[dL].[dL].P.Y}$PEPTIDE1,PEPTIDE1,1:R1-6:R2$$$',]
    # sfn = ScoringFunctions()
    # print(sfn.scores(helm_seqs, step=1))

    # sfn = ScoringFunctions(scoring_func_names=['permeability'])
    # print(sfn.scores(helm_seqs, step=1))

    aa_seqs = ["YPEDILDKHLQRVIL", "SGKVSYPEDILDKHLQRVIL","EGEKQYPEDILDKHLQRVIL","SQRPYPEDILDKHLQRVIL","QGSQPYPEDILDKHLQRVIL"]
    # docker = DockScorer()
    # print(docker._call_adcp_dock(aa_seqs))

    # helms = [create_helm_from_aa_seq(aa_seq) for aa_seq in aa_seqs]
    # sfn = ScoringFunctions(scoring_func_names=['beta_catenin'])
    # print(sfn.scores(helms, step=1))

    helms = [create_helm_from_aa_seq(aa_seq, cyclic=True) for aa_seq in aa_seqs]
    sfn = ScoringFunctions(scoring_func_names=['beta_catenin'])
    print(sfn.scores(helms, step=1))


if __name__ == "__main__":
    unit_tests()
