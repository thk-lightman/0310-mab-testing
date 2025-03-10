import numpy as np
import pandas as pd
import logging
import json
import os
from datetime import datetime
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/mab.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('mab')

class BanditAlgorithm(ABC):
    """Abstract base class for Multi-Armed Bandit algorithms"""
    
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.name = "AbstractBandit"
    
    @abstractmethod
    def select_arm(self):
        """Select which arm to pull"""
        pass
    
    def update(self, chosen_arm, reward):
        """Update the algorithm's knowledge based on reward received"""
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value
    
    def get_stats(self):
        """Return current statistics about the bandit algorithm"""
        return {
            'algorithm': self.name,
            'values': self.values.tolist(),
            'counts': self.counts.tolist(),
            'total_tries': int(np.sum(self.counts))
        }
    
    def save_state(self, file_path):
        """Save the current state of the bandit algorithm"""
        state = {
            'algorithm': self.name,
            'n_arms': self.n_arms,
            'counts': self.counts.tolist(),
            'values': self.values.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(file_path, 'w') as f:
            json.dump(state, f)
        
        logger.info(f"Saved {self.name} state to {file_path}")
    
    def load_state(self, file_path):
        """Load a previously saved state"""
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            if state['algorithm'] != self.name:
                logger.warning(f"Loading state for {state['algorithm']} into {self.name} instance")
            
            self.n_arms = state['n_arms']
            self.counts = np.array(state['counts'])
            self.values = np.array(state['values'])
            
            logger.info(f"Loaded {self.name} state from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
            return False


class EpsilonGreedy(BanditAlgorithm):
    """Epsilon-Greedy algorithm for MAB problem"""
    
    def __init__(self, n_arms, epsilon=0.1):
        super().__init__(n_arms)
        self.epsilon = epsilon
        self.name = "EpsilonGreedy"
    
    def select_arm(self):
        """Select an arm according to epsilon-greedy strategy"""
        if np.random.random() < self.epsilon:
            # Exploration: randomly select an arm
            return np.random.randint(self.n_arms)
        else:
            # Exploitation: select the arm with highest estimated value
            # If multiple arms have the same value, randomly select one of them
            return np.random.choice(np.where(self.values == np.max(self.values))[0])


class UCB1(BanditAlgorithm):
    """Upper Confidence Bound (UCB1) algorithm for MAB problem"""
    
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.name = "UCB1"
    
    def select_arm(self):
        """Select an arm according to UCB1 strategy"""
        # Pull each arm at least once
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        
        total_counts = np.sum(self.counts)
        ucb_values = self.values + np.sqrt(2 * np.log(total_counts) / self.counts)
        
        # If multiple arms have the same UCB value, randomly select one
        return np.random.choice(np.where(ucb_values == np.max(ucb_values))[0])


class ThompsonSampling(BanditAlgorithm):
    """Thompson Sampling algorithm for MAB problem (Beta distribution for Bernoulli bandits)"""
    
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.name = "ThompsonSampling"
        # For Bernoulli bandits, we track successes and failures
        self.successes = np.zeros(n_arms)
        self.failures = np.zeros(n_arms)
    
    def select_arm(self):
        """Select an arm according to Thompson Sampling strategy"""
        samples = np.zeros(self.n_arms)
        
        for arm in range(self.n_arms):
            # Sample from Beta distribution
            # Alpha = successes + 1, Beta = failures + 1 (add 1 for prior)
            if self.counts[arm] == 0:
                samples[arm] = np.random.beta(1, 1)  # Uniform prior
            else:
                samples[arm] = np.random.beta(self.successes[arm] + 1, self.failures[arm] + 1)
        
        # Select the arm with the highest sample
        return np.argmax(samples)
    
    def update(self, chosen_arm, reward):
        """Update Beta distribution parameters based on observed reward"""
        super().update(chosen_arm, reward)
        
        # Update success/failure counts
        if reward == 1:
            self.successes[chosen_arm] += 1
        else:
            self.failures[chosen_arm] += 1
    
    def get_stats(self):
        """Return current statistics including Beta distribution parameters"""
        stats = super().get_stats()
        stats.update({
            'successes': self.successes.tolist(),
            'failures': self.failures.tolist()
        })
        return stats
    
    def save_state(self, file_path):
        """Save the current state including Beta parameters"""
        state = {
            'algorithm': self.name,
            'n_arms': self.n_arms,
            'counts': self.counts.tolist(),
            'values': self.values.tolist(),
            'successes': self.successes.tolist(),
            'failures': self.failures.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(file_path, 'w') as f:
            json.dump(state, f)
        
        logger.info(f"Saved {self.name} state to {file_path}")
    
    def load_state(self, file_path):
        """Load a previously saved state including Beta parameters"""
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            if state['algorithm'] != self.name:
                logger.warning(f"Loading state for {state['algorithm']} into {self.name} instance")
            
            self.n_arms = state['n_arms']
            self.counts = np.array(state['counts'])
            self.values = np.array(state['values'])
            self.successes = np.array(state['successes'])
            self.failures = np.array(state['failures'])
            
            logger.info(f"Loaded {self.name} state from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
            return False


class WebDesignBandit:
    """MAB implementation specifically for web design testing"""
    
    def __init__(self, algorithm_name='thompson', n_designs=3, state_file=None):
        self.n_designs = n_designs
        
        # Choose algorithm
        if algorithm_name.lower() == 'epsilon':
            self.algorithm = EpsilonGreedy(n_designs)
        elif algorithm_name.lower() == 'ucb':
            self.algorithm = UCB1(n_designs)
        elif algorithm_name.lower() == 'thompson':
            self.algorithm = ThompsonSampling(n_designs)
        else:
            logger.warning(f"Unknown algorithm '{algorithm_name}', defaulting to Thompson Sampling")
            self.algorithm = ThompsonSampling(n_designs)
        
        # Try to load saved state
        if state_file and os.path.exists(state_file):
            self.algorithm.load_state(state_file)
    
    def select_design(self):
        """Select which design to show to a user"""
        selected_design = self.algorithm.select_arm()
        logger.info(f"Selected design {selected_design} using {self.algorithm.name}")
        return selected_design
    
    def record_conversion(self, design, converted):
        """Record whether the user converted (1) or not (0)"""
        reward = 1 if converted else 0
        self.algorithm.update(design, reward)
        logger.info(f"Recorded {'conversion' if converted else 'non-conversion'} for design {design}")
    
    def get_current_stats(self):
        """Get current statistics about designs"""
        stats = self.algorithm.get_stats()
        stats['design_names'] = [f"Design {i+1}" for i in range(self.n_designs)]
        return stats
    
    def save_state(self, file_path='data/bandit_state.json'):
        """Save current state to file"""
        self.algorithm.save_state(file_path)
    
    def generate_report(self, output_file='data/mab_report.json'):
        """Generate a detailed report of the MAB performance"""
        stats = self.get_current_stats()
        
        # Calculate additional metrics
        total_impressions = sum(stats['counts'])
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'algorithm': stats['algorithm'],
            'total_impressions': total_impressions,
            'designs': []
        }
        
        for i in range(self.n_designs):
            design_data = {
                'name': f"Design {i+1}",
                'impressions': int(stats['counts'][i]),
                'conversion_rate': stats['values'][i],
                'impression_percentage': stats['counts'][i] / total_impressions if total_impressions > 0 else 0
            }
            report['designs'].append(design_data)
        
        # Save report to file
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Generated MAB report and saved to {output_file}")
        return report 