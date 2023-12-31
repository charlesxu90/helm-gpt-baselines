import datetime
import time
import json
from collections import OrderedDict
from typing import List, Any, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from loguru import logger

from .goal_directed_generator import GoalDirectedGenerator
from .permeability import Permeability
from .kras import KRASInhibition
from utils.metrics_utils import Metrics

def get_time_string():
    lt = time.localtime()
    return "%04d%02d%02d-%02d%02d" % (lt.tm_year, lt.tm_mon, lt.tm_mday, lt.tm_hour, lt.tm_min)


class GoalDirectedBenchmarkResult:
    """
    Contains the results of a goal-directed benchmark.
    """

    def __init__(self, benchmark_name: str, score: float, optimized_molecules: List[Tuple[str, float]],
                 execution_time: float, number_scoring_function_calls: int, metadata: Dict[str, Any]) -> None:
        """
        Args:
            benchmark_name: name of the goal-directed benchmark
            score: benchmark score
            optimized_molecules: generated molecules, given as a list of (SMILES string, molecule score) tuples
            execution_time: execution time for the benchmark in seconds
            number_scoring_function_calls: number of calls to the scoring function
            metadata: benchmark-specific information
        """
        self.benchmark_name = benchmark_name
        self.score = score
        self.optimized_molecules = optimized_molecules
        self.execution_time = execution_time
        self.number_scoring_function_calls = number_scoring_function_calls
        self.metadata = metadata


class permeability_objective:
    def __init__(self, input_type='smiles'):
        self.scoring_func = Permeability(input_type=input_type)

    def score(self, input_seq):
        scores = self.scoring_func.get_scores([input_seq])[0]
        return scores

    def score_list(self, input_seqs):
        return self.scoring_func.get_scores(input_seqs)


class kras_objective:
    def __init__(self, input_type='smiles'):
        self.kras = KRASInhibition(input_type=input_type)

    def score(self, input_seq):
        kras_score = self.kras.trans_fn(self.kras.get_scores([input_seq]))
        return kras_score.tolist()[0]

    def score_list(self, input_seqs):
        kras_scores = self.kras.trans_fn(self.kras.get_scores(input_seqs))
        return kras_scores.tolist()

class kras_perm_objective:
    def __init__(self, input_type='smiles'):
        self.kras = KRASInhibition(input_type=input_type)
        self.perm = Permeability(input_type=input_type)

    def score(self, input_seq):
        kras_score = self.kras.trans_fn(self.kras.get_scores([input_seq]))
        perm_score = self.perm.trans_fn(self.perm.get_scores([input_seq]))
        return (kras_score + perm_score).tolist()[0] / 2

    def score_list(self, input_seqs):
        kras_scores = self.kras.trans_fn(self.kras.get_scores(input_seqs))
        perm_scores = self.perm.trans_fn(self.perm.get_scores(input_seqs))

        scores = ((kras_scores + perm_scores)/2).tolist()
        return scores

class ScoringFunctionWrapper:
    """
    Wraps a scoring function to store the number of calls to it.
    """

    def __init__(self, scoring_function) -> None:
        super().__init__()
        self.scoring_function = scoring_function
        self.evaluations = 0

    def score(self, smiles):
        self._increment_evaluation_count(1)
        return self.scoring_function.score(smiles)

    def score_list(self, smiles_list):
        self._increment_evaluation_count(len(smiles_list))
        return self.scoring_function.score_list(smiles_list)

    def _increment_evaluation_count(self, n: int):
        # Ideally, this should be protected by a lock in order to allow for multithreading.
        # However, adding a threading.Lock member variable makes the class non-pickle-able, which prevents any multithreading.
        # Therefore, in the current implementation there cannot be a guarantee that self.evaluations will be calculated correctly.
        self.evaluations += n


class GoalDirectedBenchmark:
    """
    This class assesses how well a model is able to generate molecules satisfying a given objective.
    """

    def __init__(self, name: str, objective,
                 number_molecules_to_generate=1000,
                 starting_population: Optional[List[str]] = None) -> None:
        """
        Args:
            name: Benchmark name
            objective: Objective for the goal-directed optimization
            contribution_specification: Specifies how to calculate the global benchmark score
        """
        self.name = name
        self.objective = objective
        self.wrapped_objective = ScoringFunctionWrapper(scoring_function=objective)
        self.number_molecules_to_generate = number_molecules_to_generate
        self.starting_population = starting_population
        self.metrics = Metrics(input_type='smiles')

    def assess_model(self, model: GoalDirectedGenerator) -> GoalDirectedBenchmarkResult:
        """
        Assess the given model by asking it to generate molecules optimizing a scoring function.
        The number of molecules to generate is determined automatically from the score contribution specification.

        Args:
            model: model to assess
        """
        start_time = time.time()
        molecules = model.generate_optimized_molecules(scoring_function=self.wrapped_objective,
                                                       number_molecules=self.number_molecules_to_generate,
                                                       starting_population=self.starting_population)
        end_time = time.time()
        logger.info(f'Generated {len(molecules)} molecules in {int((end_time - start_time) // 60)} min, {int((end_time - start_time) % 60)}s')

        metrics = self.metrics.get_metrics(molecules)
        scores = self.wrapped_objective.score_list(molecules)
        global_score = np.mean(scores).item()
        metrics[self.name] = global_score

        scored_molecules = zip(molecules, scores)
        sorted_scored_molecules = sorted(scored_molecules, key=lambda x: (x[1], x[0]), reverse=True)

        molecules_df = pd.DataFrame(sorted_scored_molecules, columns=['smiles', 'score'])
        molecules_df.to_csv(f'{self.name}_{model.__class__.__name__}_generated_molecules.csv', index=False)

        return GoalDirectedBenchmarkResult(benchmark_name=self.name,
                                           score=global_score,
                                           optimized_molecules=sorted_scored_molecules,
                                           execution_time=end_time - start_time,
                                           number_scoring_function_calls=self.wrapped_objective.evaluations,
                                           metadata=metrics)


def goal_directed_suite_v2() -> List[GoalDirectedBenchmark]:
    return [
        # GoalDirectedBenchmark('permeability', permeability_objective(), number_molecules_to_generate=10),
        # GoalDirectedBenchmark('permeability', permeability_objective(), number_molecules_to_generate=1000),
        # GoalDirectedBenchmark('kras', kras_objective(), number_molecules_to_generate=10),
        # GoalDirectedBenchmark('kras', kras_objective(), number_molecules_to_generate=1000),
        # GoalDirectedBenchmark('kras_perm', kras_perm_objective(), number_molecules_to_generate=10),
        GoalDirectedBenchmark('kras_perm', kras_perm_objective(), number_molecules_to_generate=1000),
    ]


def assess_goal_directed_generation(goal_directed_molecule_generator: GoalDirectedGenerator,
                                    json_output_file='output_goal_directed.json',) -> None:
    """
    Assesses a distribution-matching model for de novo molecule design.

    Args:
        goal_directed_molecule_generator: Model to evaluate
        json_output_file: Name of the file where to save the results in JSON format
        benchmark_version: which benchmark suite to execute
    """
    benchmarks = goal_directed_suite_v2()

    results = _evaluate_goal_directed_benchmarks(
        goal_directed_molecule_generator=goal_directed_molecule_generator,
        benchmarks=benchmarks)

    benchmark_results: Dict[str, Any] = OrderedDict()
    benchmark_results['timestamp'] = get_time_string()
    benchmark_results['results'] = [vars(result) for result in results]

    logger.info(f'Save results to file {json_output_file}')
    with open(json_output_file, 'wt') as f:
        f.write(json.dumps(benchmark_results, indent=4))


def _evaluate_goal_directed_benchmarks(goal_directed_molecule_generator: GoalDirectedGenerator,
                                       benchmarks: List[GoalDirectedBenchmark]
                                       ) -> List[GoalDirectedBenchmarkResult]:
    """
    Evaluate a model with the given benchmarks.
    Should not be called directly except for testing purposes.

    Args:
        goal_directed_molecule_generator: model to assess
        benchmarks: list of benchmarks to evaluate
        json_output_file: Name of the file where to save the results in JSON format
    """

    logger.info(f'Number of benchmarks: {len(benchmarks)}')

    results = []
    for i, benchmark in enumerate(benchmarks, 1):
        logger.info(f'Running benchmark {i}/{len(benchmarks)}: {benchmark.name}')
        result = benchmark.assess_model(goal_directed_molecule_generator)
        logger.info(f'Results for the benchmark:')
        logger.info(f'  Score: {result.score:.6f}')
        logger.info(f'  Execution time: {str(datetime.timedelta(seconds=int(result.execution_time)))}')
        logger.info(f'  Metadata: {result.metadata}')
        results.append(result)

    logger.info('Finished execution of the benchmarks')

    return results
