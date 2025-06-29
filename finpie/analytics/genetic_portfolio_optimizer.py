"""
Genetic Portfolio Optimizer

This module provides an object-oriented framework for optimizing portfolio strategies
using genetic algorithms. It implements proper evolutionary algorithm practices including
selection, crossover, mutation, and elitism.

Author: AI Assistant
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from copy import deepcopy
import warnings

# Import the required classes from finpie
from finpie.data.multitimeseries import MultiTimeSeries
from finpie.data.timeseries import TimeSeries


@dataclass
class OptimizationConfig:
    """Configuration class for genetic optimization parameters."""
    population_size: int = 100
    max_generations: int = 1000
    max_strategies: int = 10
    elite_size: int = 10
    tournament_size: int = 5
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    early_stopping_patience: int = 50
    convergence_threshold: float = 1e-6
    max_drawdown_threshold: float = -30000
    random_seed: Optional[int] = None
    
    # Mass extinction event parameters
    mass_extinction_events: int = 0          # Number of extinction events to trigger
    extinction_percentage: float = 0.7       # Percentage of population to kill (0.0 to 1.0)
    protect_elite_from_extinction: bool = True  # Whether elite individuals are protected from extinction


@dataclass
class FitnessMetrics:
    """Container for fitness evaluation metrics."""
    pnl: float
    max_drawdown: float
    sharpe_ratio: float
    volatility: float
    return_to_risk: float
    fitness_score: float


class Individual:
    """Represents a single solution (portfolio strategy combination) in the genetic algorithm."""
    
    def __init__(self, strategies: List[str], max_strategies: int):
        """
        Initialize an individual with a set of strategies.
        
        Args:
            strategies: List of strategy names that make up this individual
            max_strategies: Maximum number of strategies allowed
        """
        self.strategies = sorted(strategies)  # Keep strategies sorted for consistency
        self.max_strategies = max_strategies
        self.fitness_metrics: Optional[FitnessMetrics] = None
        self.age = 0  # Track how many generations this individual has survived
        
    def __eq__(self, other) -> bool:
        """Check if two individuals are identical."""
        if not isinstance(other, Individual):
            return False
        return self.strategies == other.strategies
    
    def __hash__(self) -> int:
        """Make Individual hashable for set operations."""
        return hash(tuple(self.strategies))
    
    def __repr__(self) -> str:
        """String representation of the individual."""
        fitness_str = f", fitness={self.fitness_metrics.fitness_score:.4f}" if self.fitness_metrics else ""
        return f"Individual(strategies={self.strategies[:3]}{'...' if len(self.strategies) > 3 else ''}{fitness_str})"
    
    def get_weights(self) -> Dict[str, float]:
        """Convert strategies to binary weights dictionary (1 for included, 0 for excluded)."""
        if not self.strategies:
            return {}
        return {strategy: 1.0 for strategy in self.strategies}
    
    def mutate(self, available_strategies: List[str], mutation_rate: float, rng: np.random.RandomState) -> 'Individual':
        """
        Create a mutated copy of this individual.
        
        Args:
            available_strategies: All available strategies to choose from
            mutation_rate: Probability of mutation for each gene
            rng: Random number generator
            
        Returns:
            New mutated Individual
        """
        new_strategies = self.strategies.copy()
        
        # Randomly remove strategies
        if len(new_strategies) > 1:  # Keep at least one strategy
            for i in range(len(new_strategies) - 1, -1, -1):
                if rng.random() < mutation_rate:
                    new_strategies.pop(i)
        
        # Randomly add strategies
        available_to_add = [s for s in available_strategies if s not in new_strategies]
        while (len(new_strategies) < self.max_strategies and 
               available_to_add and 
               rng.random() < mutation_rate):
            strategy_to_add = rng.choice(available_to_add)
            new_strategies.append(strategy_to_add)
            available_to_add.remove(strategy_to_add)
        
        return Individual(new_strategies, self.max_strategies)


class FitnessEvaluator:
    """Handles fitness evaluation for individuals."""
    
    def __init__(self, mts: MultiTimeSeries, config: OptimizationConfig, population_history: Dict[int, List[Individual]] = None):
        """
        Initialize fitness evaluator.
        
        Args:
            mts: MultiTimeSeries object containing strategy data
            config: Optimization configuration
            population_history: Reference to the population history for caching
        """
        self.mts = mts
        self.config = config
        self.cache: Dict[tuple, FitnessMetrics] = {}
        self.population_history = population_history or {}
    
    def evaluate(self, individual: Individual) -> FitnessMetrics:
        """
        Evaluate fitness of an individual.
        
        Args:
            individual: Individual to evaluate
            
        Returns:
            FitnessMetrics object with calculated metrics
        """
        # Check cache first
        cache_key = tuple(individual.strategies)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Check population history for identical individual
        for generation, population in self.population_history.items():
            for hist_individual in population:
                if (hist_individual.strategies == individual.strategies and 
                    hist_individual.fitness_metrics is not None):
                    # Cache this result for future use
                    self.cache[cache_key] = hist_individual.fitness_metrics
                    return hist_individual.fitness_metrics
        
        try:
            # Calculate portfolio performance
            weights = individual.get_weights()
            portfolio_ts = self.mts.portfolio(weights=weights, shares=True, percentage=False)
            portfolio_data = portfolio_ts.data * sum(weights.values())
            portfolio_ts = TimeSeries(portfolio_data)

            # Calculate metrics
            pnl = float(portfolio_data.iloc[-1].iloc[0])
            max_dd = float(portfolio_ts.max_drawdown(percentage=False).iloc[0])
            sharpe = float(portfolio_ts.sharpe_ratio(method='absolute').iloc[0])
            
            # Calculate additional metrics
            returns = portfolio_ts.returns()
            volatility = float(returns.volatility().iloc[0]) if not returns.data.empty else 0.0
            return_to_risk = abs(pnl / volatility) if volatility != 0 else 0.0
            
            # Calculate composite fitness score
            fitness_score = self._calculate_fitness_score(pnl, max_dd, sharpe, return_to_risk)
            
            metrics = FitnessMetrics(
                pnl=pnl,
                max_drawdown=max_dd,
                sharpe_ratio=sharpe,
                volatility=volatility,
                return_to_risk=return_to_risk,
                fitness_score=fitness_score
            )
            
            # Cache the result
            self.cache[cache_key] = metrics
            return metrics
            
        except Exception as e:
            # Return poor fitness for invalid individuals
            warnings.warn(f"Error evaluating individual {individual}: {e}")
            return FitnessMetrics(
                pnl=-1e6,
                max_drawdown=-1e6,
                sharpe_ratio=-1e6,
                volatility=1e6,
                return_to_risk=0.0,
                fitness_score=-1e6
            )
    
    def _calculate_fitness_score(self, pnl: float, max_drawdown: float, 
                                sharpe_ratio: float, return_to_risk: float) -> float:
        """
        Calculate composite fitness score.
        
        Args:
            pnl: Profit and loss
            max_drawdown: Maximum drawdown
            sharpe_ratio: Sharpe ratio
            return_to_risk: Return to risk ratio
            
        Returns:
            Composite fitness score
        """
        # Weighted combination of metrics
        fitness = pnl * (max_drawdown > self.config.max_drawdown_threshold)
        
        return float(fitness)


class GeneticPortfolioOptimizer:
    """Main genetic algorithm optimizer for portfolio strategy selection."""
    
    def __init__(self, mts: MultiTimeSeries, config: OptimizationConfig = None):
        """
        Initialize the genetic optimizer.
        
        Args:
            mts: MultiTimeSeries object containing strategy data
            config: Optimization configuration
        """
        self.mts = mts
        self.config = config or OptimizationConfig()
        self.available_strategies = list(mts.data.columns)
        
        # Initialize random number generator
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        self.rng = np.random.RandomState(self.config.random_seed)
        
        # Initialize state
        self.population: List[Individual] = []
        self.population_history: Dict[int, List[Individual]] = {}  # Complete history of all generations
        self.best_individual = None
        self.generation = 0
        self.optimization_history = []
        
        # Mass extinction event tracking
        self.extinction_events_occurred: int = 0
        self.extinction_history: List[Dict[str, Any]] = []
        self.extinction_generations: List[int] = []  # Track generations where extinctions occurred

        # Initialize components
        self.fitness_evaluator = FitnessEvaluator(mts, self.config, self.population_history)
        
        # Setup logging
        #logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _create_random_individual(self) -> Individual:
        """Create a random individual."""
        num_strategies = self.rng.randint(1, self.config.max_strategies + 1)
        strategies = self.rng.choice(
            self.available_strategies, 
            size=num_strategies, 
            replace=False
        ).tolist()
        return Individual(strategies, self.config.max_strategies)
    
    def _initialize_population(self) -> List[Individual]:
        """Initialize the population with random individuals."""
        population = []
        seen_individuals = set()
        
        # Create diverse initial population
        attempts = 0
        max_attempts = self.config.population_size * 10
        
        while len(population) < self.config.population_size and attempts < max_attempts:
            individual = self._create_random_individual()
            if individual not in seen_individuals:
                population.append(individual)
                seen_individuals.add(individual)
            attempts += 1
        
        # Fill remaining slots if needed
        while len(population) < self.config.population_size:
            individual = self._create_random_individual()
            population.append(individual)
        
        return population
    
    def _evaluate_population(self, population: List[Individual]):
        """Evaluate fitness for all individuals in population."""
        for individual in population:
            if individual.fitness_metrics is None:
                individual.fitness_metrics = self.fitness_evaluator.evaluate(individual)
    
    def _tournament_selection(self, population: List[Individual], k: int) -> List[Individual]:
        """Select k individuals using tournament selection."""
        selected = []
        for _ in range(k):
            tournament_size = min(self.config.tournament_size, len(population))
            tournament = self.rng.choice(population, size=tournament_size, replace=False)
            winner = max(tournament, key=lambda ind: ind.fitness_metrics.fitness_score)
            selected.append(winner)
        return selected
    
    def _uniform_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Create offspring using uniform crossover."""
        all_strategies = list(set(parent1.strategies + parent2.strategies))
        max_strategies = parent1.max_strategies
        
        # Create first offspring
        child1_strategies = []
        for strategy in all_strategies:
            if len(child1_strategies) < max_strategies:
                # Choose from parent that has this strategy, or random if both have it
                if strategy in parent1.strategies and strategy in parent2.strategies:
                    if self.rng.random() < 0.5:
                        child1_strategies.append(strategy)
                elif strategy in parent1.strategies:
                    if self.rng.random() < 0.7:  # Bias towards including parent strategies
                        child1_strategies.append(strategy)
                elif strategy in parent2.strategies:
                    if self.rng.random() < 0.7:
                        child1_strategies.append(strategy)
        
        # Ensure minimum strategies
        if not child1_strategies:
            child1_strategies = [self.rng.choice(all_strategies)]
        
        # Create second offspring with complementary strategies
        remaining_strategies = [s for s in all_strategies if s not in child1_strategies]
        child2_strategies = []
        
        # Add some strategies from child1 to child2
        for strategy in child1_strategies:
            if len(child2_strategies) < max_strategies and self.rng.random() < 0.3:
                child2_strategies.append(strategy)
        
        # Add remaining strategies
        for strategy in remaining_strategies:
            if len(child2_strategies) < max_strategies and self.rng.random() < 0.7:
                child2_strategies.append(strategy)
        
        # Ensure minimum strategies
        if not child2_strategies:
            child2_strategies = [self.rng.choice(all_strategies)]
        
        return (Individual(child1_strategies, max_strategies), 
                Individual(child2_strategies, max_strategies))
    
    def _create_next_generation(self, population: List[Individual]) -> List[Individual]:
        """Create the next generation using selection, crossover, and mutation."""
        next_population = []
        
        # Sort population by fitness
        sorted_pop = sorted(population, 
                          key=lambda ind: ind.fitness_metrics.fitness_score, 
                          reverse=True)
        
        # Elitism: carry over best individuals
        elite = sorted_pop[:self.config.elite_size]
        for individual in elite:
            individual.age += 1
            next_population.append(deepcopy(individual))
        
        # Generate offspring
        while len(next_population) < self.config.population_size:
            # Selection
            parents = self._tournament_selection(population, 2)
            parent1, parent2 = parents[0], parents[1]
            
            # Crossover
            if self.rng.random() < self.config.crossover_rate:
                child1, child2 = self._uniform_crossover(parent1, parent2)
            else:
                child1, child2 = deepcopy(parent1), deepcopy(parent2)
            
            # Mutation
            if self.rng.random() < self.config.mutation_rate:
                child1 = child1.mutate(self.available_strategies, self.config.mutation_rate, self.rng)
            if self.rng.random() < self.config.mutation_rate:
                child2 = child2.mutate(self.available_strategies, self.config.mutation_rate, self.rng)
            
            # Add to next generation
            if len(next_population) < self.config.population_size:
                next_population.append(child1)
            if len(next_population) < self.config.population_size:
                next_population.append(child2)
        
        return next_population[:self.config.population_size]
    
    def _trigger_mass_extinction_event(self, generation: int) -> List[Individual]:
        """
        Trigger a mass extinction event that kills a percentage of the population.
        
        Args:
            generation: Current generation number
            
        Returns:
            New population after extinction event
        """
        original_population = deepcopy(self.population)
        population_size = len(self.population)
        extinction_count = int(population_size * self.config.extinction_percentage)
        survivors_count = population_size - extinction_count
        
        # Sort population by fitness
        sorted_population = sorted(self.population, 
                                 key=lambda ind: ind.fitness_metrics.fitness_score, 
                                 reverse=True)
        
        survivors = []
        
        if self.config.protect_elite_from_extinction:
            # Protect elite from extinction
            elite_count = min(self.config.elite_size, survivors_count)
            survivors.extend(sorted_population[:elite_count])
            
            # Randomly select remaining survivors from the rest
            remaining_population = sorted_population[elite_count:]
            remaining_survivors_needed = survivors_count - elite_count
            
            if remaining_survivors_needed > 0 and remaining_population:
                random_survivors = self.rng.choice(
                    remaining_population, 
                    size=min(remaining_survivors_needed, len(remaining_population)), 
                    replace=False
                ).tolist()
                survivors.extend(random_survivors)
        else:
            # No protection - randomly select survivors from entire population
            survivors = self.rng.choice(
                self.population, 
                size=survivors_count, 
                replace=False
            ).tolist()
        
        # Create new random individuals to replace the extinct ones
        new_individuals = []
        for _ in range(extinction_count):
            new_individuals.append(self._create_random_individual())
        
        # Combine survivors and new individuals
        new_population = survivors + new_individuals
        
        # Evaluate new individuals
        for individual in new_individuals:
            individual.fitness_metrics = self.fitness_evaluator.evaluate(individual)
        
        # Log extinction event
        extinction_event = {
            'generation': generation,
            'event_number': self.extinction_events_occurred + 1,
            'population_size': population_size,
            'extinction_count': extinction_count,
            'survivors_count': survivors_count,
            'elite_protected': self.config.protect_elite_from_extinction,
            'best_fitness_before': max(ind.fitness_metrics.fitness_score for ind in original_population),
            'avg_fitness_before': np.mean([ind.fitness_metrics.fitness_score for ind in original_population]),
            'best_fitness_after': max(ind.fitness_metrics.fitness_score for ind in new_population),
            'avg_fitness_after': np.mean([ind.fitness_metrics.fitness_score for ind in new_population])
        }
        
        self.extinction_history.append(extinction_event)
        self.extinction_events_occurred += 1
        self.extinction_generations.append(generation)  # Track the generation
        
        # Log the extinction event
        print(f"🌋 MASS EXTINCTION EVENT #{self.extinction_events_occurred} at generation {generation}")
        print(f"   💀 Killed: {extinction_count} individuals ({self.config.extinction_percentage:.1%})")
        print(f"   🛡️  Survivors: {survivors_count} individuals")
        print(f"   🧬 New individuals: {extinction_count}")
        print(f"   📊 Elite protected: {self.config.protect_elite_from_extinction}")
        
        return new_population
    
    def optimize(self) -> Tuple[Individual, pd.DataFrame]:
        """
        Run the genetic optimization algorithm.
        
        Returns:
            Tuple of (best_individual, optimization_history)
        """
        print("Starting genetic optimization...")
        print(f"Population size: {self.config.population_size}")
        print(f"Max generations: {self.config.max_generations}")
        print(f"Available strategies: {len(self.available_strategies)}")
        
        # Initialize population
        self.population = self._initialize_population()
        self._evaluate_population(self.population)
        
        # Store initial population in history
        self.population_history[0] = deepcopy(self.population)
        
        # Track best fitness for convergence
        best_fitness_history = []
        convergence_counter = 0
        
        # Evolution loop
        for generation in range(self.config.max_generations):
            self.generation = generation
            
            # Get best individual
            current_best = max(self.population, key=lambda ind: ind.fitness_metrics.fitness_score)
            
            if self.best_individual is None or (
                current_best.fitness_metrics.fitness_score > 
                self.best_individual.fitness_metrics.fitness_score
            ):
                self.best_individual = deepcopy(current_best)
                convergence_counter = 0
            else:
                convergence_counter += 1
            
            # Track fitness history
            best_fitness_history.append(current_best.fitness_metrics.fitness_score)
            
            # Log progress
            if generation % 10 == 0 or generation < 10:
                avg_fitness = np.mean([ind.fitness_metrics.fitness_score for ind in self.population])
                print(
                    f"Generation {generation}: "
                    f"Best fitness: {current_best.fitness_metrics.fitness_score:.2f}, "
                    f"Avg fitness: {avg_fitness:.2f}, "
                    f"Best PnL: {current_best.fitness_metrics.pnl:.2f}, "
                    f"Best strategies: {len(current_best.strategies)}"
                )
            
            # Record history
            self.optimization_history.append({
                'generation': generation,
                'best_fitness': current_best.fitness_metrics.fitness_score,
                'best_pnl': current_best.fitness_metrics.pnl,
                'best_max_drawdown': current_best.fitness_metrics.max_drawdown,
                'best_sharpe_ratio': current_best.fitness_metrics.sharpe_ratio,
                'avg_fitness': np.mean([ind.fitness_metrics.fitness_score for ind in self.population]),
                'population_diversity': len(set(tuple(ind.strategies) for ind in self.population)),
                'num_strategies': len(current_best.strategies),
                'strategies': current_best.strategies.copy(),
                'extinction_events_occurred': self.extinction_events_occurred,
                'is_extinction_generation': False  # Will be updated if extinction occurs
            })
            
            # Check convergence and mass extinction events
            should_converge = False
            
            if len(best_fitness_history) >= self.config.early_stopping_patience:
                recent_improvement = (
                    max(best_fitness_history[-self.config.early_stopping_patience:]) - 
                    min(best_fitness_history[-self.config.early_stopping_patience:])
                )
                if recent_improvement < self.config.convergence_threshold:
                    should_converge = True
            
            # Early stopping based on no improvement
            if convergence_counter >= self.config.early_stopping_patience:
                should_converge = True
            
            # Handle convergence and mass extinction events
            if should_converge:
                remaining_extinctions = self.config.mass_extinction_events - self.extinction_events_occurred
                
                if remaining_extinctions > 0:
                    # Trigger mass extinction instead of stopping
                    print(f"Convergence detected, but {remaining_extinctions} extinction events remaining")
                    self.population = self._trigger_mass_extinction_event(generation)
                    
                    # Mark this generation as an extinction generation
                    if self.optimization_history:
                        self.optimization_history[-1]['is_extinction_generation'] = True
                    
                    # Reset convergence tracking after extinction
                    convergence_counter = 0
                    best_fitness_history = best_fitness_history[-10:]  # Keep some history but reset tracking
                    
                    # Continue to next generation after extinction
                    continue
                else:
                    # All extinction events used, can now stop
                    if len(best_fitness_history) >= self.config.early_stopping_patience and recent_improvement < self.config.convergence_threshold:
                        print(f"Converged at generation {generation} (all {self.config.mass_extinction_events} extinction events completed)")
                    else:
                        print(f"Early stopping at generation {generation} (all {self.config.mass_extinction_events} extinction events completed)")
                    break
            
            # Create next generation
            self.population = self._create_next_generation(self.population)
            self._evaluate_population(self.population)
            
            # Store generation in history
            self.population_history[generation + 1] = deepcopy(self.population)
        
        # Final results
        print("Optimization completed!")
        print(f"Best individual: {self.best_individual}")
        print(f"Best fitness: {self.best_individual.fitness_metrics.fitness_score:.4f}")
        print(f"Best PnL: {self.best_individual.fitness_metrics.pnl:.2f}")
        print(f"Best max drawdown: {self.best_individual.fitness_metrics.max_drawdown:.2f}")
        print(f"Best Sharpe ratio: {self.best_individual.fitness_metrics.sharpe_ratio:.4f}")
        
        return self.best_individual, pd.DataFrame(self.optimization_history)
    
    def get_population_summary(self) -> pd.DataFrame:
        """Get summary statistics of current population."""
        if not self.population:
            return pd.DataFrame()
        
        data = []
        for i, individual in enumerate(self.population):
            if individual.fitness_metrics:
                data.append({
                    'individual_id': i,
                    'fitness_score': individual.fitness_metrics.fitness_score,
                    'pnl': individual.fitness_metrics.pnl,
                    'max_drawdown': individual.fitness_metrics.max_drawdown,
                    'sharpe_ratio': individual.fitness_metrics.sharpe_ratio,
                    'volatility': individual.fitness_metrics.volatility,
                    'num_strategies': len(individual.strategies),
                    'strategies': individual.strategies.copy(),
                    'age': individual.age
                })
        
        return pd.DataFrame(data).sort_values('fitness_score', ascending=False)
    
    def get_generation_population(self, generation: int) -> List[Individual]:
        """
        Get the population from a specific generation.
        
        Args:
            generation: Generation number to retrieve
            
        Returns:
            List of individuals from that generation
        """
        if generation not in self.population_history:
            raise ValueError(f"Generation {generation} not found in history. Available: {list(self.population_history.keys())}")
        return self.population_history[generation]
    
    def get_population_history_summary(self) -> pd.DataFrame:
        """
        Get a summary of all generations in the population history.
        
        Returns:
            DataFrame with statistics for each generation
        """
        history_data = []
        
        for generation, population in self.population_history.items():
            # Calculate generation statistics
            fitness_scores = [ind.fitness_metrics.fitness_score for ind in population if ind.fitness_metrics]
            pnls = [ind.fitness_metrics.pnl for ind in population if ind.fitness_metrics]
            strategy_counts = [len(ind.strategies) for ind in population]
            
            if fitness_scores:  # Only add if we have fitness data
                history_data.append({
                    'generation': generation,
                    'best_fitness': max(fitness_scores),
                    'avg_fitness': np.mean(fitness_scores),
                    'worst_fitness': min(fitness_scores),
                    'best_pnl': max(pnls),
                    'avg_pnl': np.mean(pnls),
                    'avg_strategies': np.mean(strategy_counts),
                    'diversity': len(set(tuple(ind.strategies) for ind in population)),
                    'population_size': len(population)
                })
        
        return pd.DataFrame(history_data)
    
    def get_individual_lineage(self, target_individual: Individual, max_generations: int = None) -> List[Tuple[int, Individual]]:
        """
        Find all individuals with the same strategy combination across generations.
        
        Args:
            target_individual: Individual to trace
            max_generations: Maximum generations to search (None for all)
            
        Returns:
            List of (generation, individual) tuples with matching strategies
        """
        lineage = []
        target_strategies = set(target_individual.strategies)
        
        search_generations = self.population_history.keys()
        if max_generations is not None:
            search_generations = [g for g in search_generations if g <= max_generations]
        
        for generation in search_generations:
            for individual in self.population_history[generation]:
                if set(individual.strategies) == target_strategies:
                    lineage.append((generation, deepcopy(individual)))
        
        return lineage
    
    def get_strategy_frequency_history(self) -> pd.DataFrame:
        """
        Get the frequency of each strategy across all generations.
        
        Returns:
            DataFrame with strategy frequencies by generation
        """
        strategy_data = []
        
        for generation, population in self.population_history.items():
            strategy_counts = {}
            
            # Count occurrences of each strategy
            for individual in population:
                for strategy in individual.strategies:
                    strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            # Add to data
            for strategy, count in strategy_counts.items():
                strategy_data.append({
                    'generation': generation,
                    'strategy': strategy,
                    'count': count,
                    'frequency': count / len(population)
                })
        
        return pd.DataFrame(strategy_data)
    
    def clear_history_before_generation(self, generation: int):
        """
        Clear population history before a specific generation to save memory.
        
        Args:
            generation: Keep history from this generation onwards
        """
        generations_to_remove = [g for g in self.population_history.keys() if g < generation]
        for g in generations_to_remove:
            del self.population_history[g]
        
        print(f"Cleared history for {len(generations_to_remove)} generations before generation {generation}")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the fitness evaluation cache.
        
        Returns:
            Dictionary with cache statistics
        """
        total_individuals = sum(len(pop) for pop in self.population_history.values())
        unique_strategies = len(self.fitness_evaluator.cache)
        
        return {
            'total_individuals_evaluated': total_individuals,
            'unique_strategy_combinations': unique_strategies,
            'cache_hit_rate': 1.0 - (unique_strategies / total_individuals) if total_individuals > 0 else 0.0,
            'generations_stored': len(self.population_history),
            'cache_size': len(self.fitness_evaluator.cache)
        }

    def get_cache_values(self) -> pd.DataFrame:
        """
        Get the values of the cache.
        
        Returns:
            DataFrame with cache values
        """
        cache_data = []
        for key, value in self.fitness_evaluator.cache.items():
            cache_data.append({
                'key': key,
                'fitness_score': value.fitness_score,
                'pnl': value.pnl,
                'max_drawdown': value.max_drawdown,
                'sharpe_ratio': value.sharpe_ratio,
                'volatility': value.volatility,
                'return_to_risk': value.return_to_risk
            })
        return pd.DataFrame(cache_data)
    
    def get_extinction_history(self) -> pd.DataFrame:
        """
        Get the history of mass extinction events.
        
        Returns:
            DataFrame with extinction event details
        """
        if not self.extinction_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.extinction_history)
    
    def get_extinction_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about mass extinction events.
        
        Returns:
            Dictionary with extinction statistics
        """
        if not self.extinction_history:
            return {
                'total_extinction_events': 0,
                'extinction_events_configured': self.config.mass_extinction_events,
                'events_remaining': self.config.mass_extinction_events,
                'avg_fitness_impact': 0.0,
                'total_individuals_killed': 0
            }
        
        fitness_impacts = []
        total_killed = 0
        
        for event in self.extinction_history:
            fitness_impact = event['avg_fitness_after'] - event['avg_fitness_before']
            fitness_impacts.append(fitness_impact)
            total_killed += event['extinction_count']
        
        return {
            'total_extinction_events': len(self.extinction_history),
            'extinction_events_configured': self.config.mass_extinction_events,
            'events_remaining': self.config.mass_extinction_events - self.extinction_events_occurred,
            'avg_fitness_impact': np.mean(fitness_impacts),
            'total_individuals_killed': total_killed,
            'extinction_generations': [event['generation'] for event in self.extinction_history],
            'avg_extinction_percentage': self.config.extinction_percentage,
            'elite_protection_enabled': self.config.protect_elite_from_extinction
        }
    
    def plot_extinction_impact(self) -> None:
        """
        Plot the impact of extinction events on population fitness.
        Requires matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for plotting")
            return
        
        if not self.extinction_history:
            print("No extinction events to plot")
            return
        
        # Get optimization history
        history_df = pd.DataFrame(self.optimization_history)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot fitness evolution with extinction markers
        ax1.plot(history_df['generation'], history_df['best_fitness'], 'b-', linewidth=2, label='Best Fitness')
        ax1.plot(history_df['generation'], history_df['avg_fitness'], 'r--', alpha=0.7, label='Avg Fitness')
        
        # Mark extinction events
        extinction_gens = [event['generation'] for event in self.extinction_history]
        for i, gen in enumerate(extinction_gens):
            ax1.axvline(x=gen, color='red', linestyle=':', alpha=0.8, linewidth=2)
            ax1.text(gen, ax1.get_ylim()[1] * 0.9, f'💀 #{i+1}', rotation=90, 
                    verticalalignment='top', horizontalalignment='right', fontsize=10)
        
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness Score')
        ax1.set_title('🧬 Fitness Evolution with Mass Extinction Events')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot population diversity
        ax2.plot(history_df['generation'], history_df['population_diversity'], 'm-', linewidth=2, label='Population Diversity')
        
        # Mark extinction events
        for i, gen in enumerate(extinction_gens):
            ax2.axvline(x=gen, color='red', linestyle=':', alpha=0.8, linewidth=2)
        
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Unique Individuals')
        ax2.set_title('🌟 Population Diversity with Extinction Events')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_post_extinction_recovery(self, recovery_generations: int = 10) -> pd.DataFrame:
        """
        Analyze how the population recovered after each extinction event.
        
        Args:
            recovery_generations: Number of generations to analyze after each extinction
            
        Returns:
            DataFrame with recovery analysis
        """
        if not self.extinction_history:
            return pd.DataFrame()
        
        history_df = pd.DataFrame(self.optimization_history)
        recovery_data = []
        
        for event in self.extinction_history:
            extinction_gen = event['generation']
            
            # Get generations after extinction
            post_extinction = history_df[
                (history_df['generation'] > extinction_gen) & 
                (history_df['generation'] <= extinction_gen + recovery_generations)
            ]
            
            if not post_extinction.empty:
                recovery_data.append({
                    'extinction_event': event['event_number'],
                    'extinction_generation': extinction_gen,
                    'fitness_immediately_after': event['avg_fitness_after'],
                    'fitness_drop': event['avg_fitness_before'] - event['avg_fitness_after'],
                    'best_fitness_in_recovery': post_extinction['best_fitness'].max(),
                    'generations_to_recover': len(post_extinction),
                    'diversity_after_extinction': post_extinction.iloc[0]['population_diversity'] if len(post_extinction) > 0 else 0,
                    'final_diversity': post_extinction.iloc[-1]['population_diversity'] if len(post_extinction) > 0 else 0
                })
        
        return pd.DataFrame(recovery_data)
    
    def get_extinction_generations(self) -> List[int]:
        """
        Get the list of generations where mass extinction events occurred.
        
        Returns:
            List of generation numbers where extinctions happened
        """
        return self.extinction_generations.copy()
    
    def plot_optimization_history(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot the optimization history with vertical lines marking extinction events.
        Requires matplotlib.
        
        Args:
            figsize: Figure size as (width, height)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for plotting")
            return
        
        if not self.optimization_history:
            print("No optimization history to plot")
            return
        
        # Get optimization history
        history = pd.DataFrame(self.optimization_history)
        
        # Create the plot
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('🧬 Genetic Optimization Progress', fontsize=16)
        
        # Fitness evolution
        axes[0, 0].plot(history['generation'], history['best_fitness'], 'b-', linewidth=2, label='Best Fitness')
        axes[0, 0].plot(history['generation'], history['avg_fitness'], 'r--', alpha=0.7, label='Avg Fitness')
        
        # Add extinction vertical lines
        for i, gen in enumerate(self.extinction_generations):
            axes[0, 0].axvline(x=gen, color='red', linestyle=':', alpha=0.8, linewidth=2)
            axes[0, 0].text(gen, axes[0, 0].get_ylim()[1] * 0.95, f'💀 #{i+1}', rotation=90, 
                           verticalalignment='top', horizontalalignment='right', fontsize=9, color='darkred')
        
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Fitness Score')
        axes[0, 0].set_title('💪 Fitness Evolution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # PnL evolution
        axes[0, 1].plot(history['generation'], history['best_pnl'], 'g-', linewidth=2)
        
        # Add extinction vertical lines
        for gen in self.extinction_generations:
            axes[0, 1].axvline(x=gen, color='red', linestyle=':', alpha=0.8, linewidth=2)
        
        axes[0, 1].set_xlabel('Generation')
        axes[0, 1].set_ylabel('PnL')
        axes[0, 1].set_title('💰 Best PnL Evolution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Population diversity
        axes[1, 0].plot(history['generation'], history['population_diversity'], 'm-', linewidth=2)
        
        # Add extinction vertical lines
        for gen in self.extinction_generations:
            axes[1, 0].axvline(x=gen, color='red', linestyle=':', alpha=0.8, linewidth=2)
        
        axes[1, 0].set_xlabel('Generation')
        axes[1, 0].set_ylabel('Unique Individuals')
        axes[1, 0].set_title('🌟 Population Diversity')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Strategy count evolution
        axes[1, 1].plot(history['generation'], history['num_strategies'], 'c-', linewidth=2)
        
        # Add extinction vertical lines
        for gen in self.extinction_generations:
            axes[1, 1].axvline(x=gen, color='red', linestyle=':', alpha=0.8, linewidth=2)
        
        axes[1, 1].set_xlabel('Generation')
        axes[1, 1].set_ylabel('Number of Strategies')
        axes[1, 1].set_title('📊 Best Individual Strategy Count')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print extinction summary
        if self.extinction_generations:
            print(f"\n🌋 EXTINCTION EVENTS SUMMARY:")
            print(f"   Total events: {len(self.extinction_generations)}")
            print(f"   Occurred at generations: {self.extinction_generations}")
            print(f"   Kill percentage: {self.config.extinction_percentage:.1%}")
            print(f"   Elite protection: {self.config.protect_elite_from_extinction}")
        else:
            print(f"\n📊 No extinction events occurred during optimization.")
    
    def plot_fitness_with_extinctions(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Simple plot focusing just on fitness evolution with extinction markers.
        
        Args:
            figsize: Figure size as (width, height)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for plotting")
            return
        
        if not self.optimization_history:
            print("No optimization history to plot")
            return
        
        history_df = pd.DataFrame(self.optimization_history)
        
        plt.figure(figsize=figsize)
        
        # Plot fitness lines
        plt.plot(history_df['generation'], history_df['best_fitness'], 'b-', linewidth=2, label='Best Fitness')
        plt.plot(history_df['generation'], history_df['avg_fitness'], 'r--', alpha=0.7, linewidth=1.5, label='Average Fitness')
        
        # Add extinction vertical lines
        for i, gen in enumerate(self.extinction_generations):
            plt.axvline(x=gen, color='red', linestyle=':', alpha=0.8, linewidth=2, 
                       label='Mass Extinction' if i == 0 else "")
            
            # Add extinction markers
            plt.text(gen, plt.ylim()[1] * (0.95 - i * 0.05), f'💀 #{i+1}', rotation=90, 
                    verticalalignment='top', horizontalalignment='right', 
                    fontsize=10, color='darkred', fontweight='bold')
        
        plt.xlabel('Generation')
        plt.ylabel('Fitness Score')
        plt.title('🧬 Fitness Evolution with Mass Extinction Events 🌋')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add text box with extinction info
        if self.extinction_generations:
            info_text = f"Extinctions: {self.extinction_generations}\nKill Rate: {self.config.extinction_percentage:.1%}"
            plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
        
        plt.tight_layout()
        plt.show()

# Example usage function
def run_optimization_example(mts: MultiTimeSeries) -> Tuple[Individual, pd.DataFrame]:
    """
    Example function showing how to use the genetic optimizer.
    
    Args:
        mts: MultiTimeSeries object containing strategy data
        
    Returns:
        Tuple of (best_individual, optimization_history)
    """
    # Configure optimization
    config = OptimizationConfig(
        population_size=50,          # Smaller for faster execution
        max_generations=100,         # Fewer generations for demo
        max_strategies=8,            # Limit strategies per portfolio
        elite_size=5,               # Keep best 5 individuals
        tournament_size=3,          # Tournament selection size
        crossover_rate=0.8,         # 80% crossover probability
        mutation_rate=0.1,          # 10% mutation probability
        early_stopping_patience=20, # Stop if no improvement for 20 generations
        random_seed=42              # For reproducible results
    )
    
    # Create and run optimizer
    optimizer = GeneticPortfolioOptimizer(mts, config)
    best_individual, history = optimizer.optimize()
    
    # Print results
    print(f"\nOptimization Results:")
    print(f"Best strategies: {best_individual.strategies}")
    print(f"Best PnL: {best_individual.fitness_metrics.pnl:.2f}")
    print(f"Best Max Drawdown: {best_individual.fitness_metrics.max_drawdown:.2f}")
    print(f"Best Sharpe Ratio: {best_individual.fitness_metrics.sharpe_ratio:.4f}")
    
    return best_individual, history 