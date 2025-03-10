"""
Contextual Bandit implementation for MAB Web Testing.
This is an extension of the basic MAB algorithms that takes user context into account.
"""

import numpy as np
import pandas as pd
import logging
import json
import os
from datetime import datetime
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/contextual_bandit.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('contextual_bandit')


class LinUCB:
    """
    LinUCB algorithm for contextual bandit problems.
    Uses a linear model to estimate rewards based on context features.
    """
    
    def __init__(self, n_arms, d, alpha=1.0):
        """
        Initialize LinUCB algorithm.
        
        Args:
            n_arms (int): Number of arms (design variants)
            d (int): Dimension of context features
            alpha (float): Exploration parameter that controls the confidence bound
        """
        self.n_arms = n_arms
        self.d = d  # dimension of context features
        self.alpha = alpha
        self.name = "LinUCB"
        
        # Initialize model parameters for each arm
        self.A = [np.identity(d) for _ in range(n_arms)]  # A_a matrices (d x d)
        self.b = [np.zeros((d, 1)) for _ in range(n_arms)]  # b_a vectors (d x 1)
        self.theta = [np.zeros((d, 1)) for _ in range(n_arms)]  # theta_a parameters (d x 1)
        
        # For tracking performance
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
    
    def select_arm(self, context):
        """
        Select an arm based on context features using LinUCB algorithm.
        
        Args:
            context: Array of context features (d-dimensional)
        
        Returns:
            int: Index of the selected arm
        """
        context = np.array(context).reshape(-1, 1)  # reshape to column vector
        
        # Calculate UCB for each arm
        ucb_values = np.zeros(self.n_arms)
        
        for arm in range(self.n_arms):
            # Compute theta_a = A_a^(-1) * b_a
            self.theta[arm] = np.linalg.solve(self.A[arm], self.b[arm])
            
            # Calculate UCB: x^T * theta_a + alpha * sqrt(x^T * A_a^(-1) * x)
            x_transpose_theta = np.dot(context.T, self.theta[arm])[0, 0]
            
            # Calculate confidence interval
            a_inv = np.linalg.inv(self.A[arm])
            confidence = self.alpha * np.sqrt(np.dot(np.dot(context.T, a_inv), context))[0, 0]
            
            # Store UCB value
            ucb_values[arm] = x_transpose_theta + confidence
        
        # Select arm with highest UCB
        return np.argmax(ucb_values)
    
    def update(self, chosen_arm, context, reward):
        """
        Update the model with the observed reward.
        
        Args:
            chosen_arm (int): Index of the chosen arm
            context: Context features associated with the action
            reward (float): Observed reward
        """
        context = np.array(context).reshape(-1, 1)  # reshape to column vector
        
        # Update counts
        self.counts[chosen_arm] += 1
        
        # Update A_a and b_a for the chosen arm
        self.A[chosen_arm] += np.dot(context, context.T)
        self.b[chosen_arm] += reward * context
        
        # Update average reward (for tracking purposes)
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
            'd': self.d,
            'alpha': self.alpha,
            'counts': self.counts.tolist(),
            'values': self.values.tolist(),
            'A': [A_a.tolist() for A_a in self.A],
            'b': [b_a.tolist() for b_a in self.b],
            'theta': [theta_a.tolist() for theta_a in self.theta],
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
            self.d = state['d']
            self.alpha = state['alpha']
            self.counts = np.array(state['counts'])
            self.values = np.array(state['values'])
            self.A = [np.array(A_a) for A_a in state['A']]
            self.b = [np.array(b_a) for b_a in state['b']]
            self.theta = [np.array(theta_a) for theta_a in state['theta']]
            
            logger.info(f"Loaded {self.name} state from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
            return False


class LogisticContextualBandit:
    """
    Contextual bandit using logistic regression models.
    Suitable for binary rewards (e.g., conversion/no conversion).
    """
    
    def __init__(self, n_arms, exploration_factor=0.1):
        """
        Initialize Logistic Contextual Bandit.
        
        Args:
            n_arms (int): Number of arms (design variants)
            exploration_factor (float): Controls exploration vs exploitation
        """
        self.n_arms = n_arms
        self.exploration_factor = exploration_factor
        self.name = "LogisticContextualBandit"
        
        # Create a logistic regression model for each arm
        self.models = [
            Pipeline([
                ('scaler', StandardScaler()),
                ('logreg', LogisticRegression(solver='lbfgs', warm_start=True))
            ]) for _ in range(n_arms)
        ]
        
        # Initialize counters and data storage for each arm
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.X = [[] for _ in range(n_arms)]  # features for each arm
        self.y = [[] for _ in range(n_arms)]  # rewards for each arm
        
        # Minimum number of samples before we start using the model
        self.min_samples = 10
    
    def select_arm(self, context):
        """
        Select an arm based on context features.
        
        Args:
            context: Array of context features
        
        Returns:
            int: Index of the selected arm
        """
        # If we don't have enough data for some arms, prefer those
        for arm in range(self.n_arms):
            if self.counts[arm] < self.min_samples:
                return arm
        
        # All arms have minimum samples, use models to predict
        scores = np.zeros(self.n_arms)
        
        for arm in range(self.n_arms):
            # Get predicted probability of conversion
            if len(self.X[arm]) > 0:
                X_arm = np.array(self.X[arm])
                y_arm = np.array(self.y[arm])
                
                try:
                    # Reshape context if it's a single sample
                    context_reshaped = np.array(context).reshape(1, -1)
                    
                    # Fit model and predict
                    self.models[arm].fit(X_arm, y_arm)
                    pred_prob = self.models[arm].predict_proba(context_reshaped)[0, 1]
                    
                    # Add exploration bonus
                    uncertainty = self.exploration_factor / np.sqrt(self.counts[arm])
                    scores[arm] = pred_prob + uncertainty
                except Exception as e:
                    logger.error(f"Error predicting for arm {arm}: {str(e)}")
                    scores[arm] = 0.5  # Default to 0.5 probability
            else:
                scores[arm] = 0.5  # Default score when no data
        
        # Choose arm with highest score
        return np.argmax(scores)
    
    def update(self, chosen_arm, context, reward):
        """
        Update the model with the observed reward.
        
        Args:
            chosen_arm (int): Index of the chosen arm
            context: Context features associated with the action
            reward (float): Observed reward (0 or 1 for binary outcomes)
        """
        # Update counts
        self.counts[chosen_arm] += 1
        
        # Store context and reward for the chosen arm
        self.X[chosen_arm].append(context)
        self.y[chosen_arm].append(reward)
        
        # Update average reward
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value
        
        # Re-train model for the chosen arm if we have enough samples
        if n >= self.min_samples:
            try:
                X_arm = np.array(self.X[chosen_arm])
                y_arm = np.array(self.y[chosen_arm])
                self.models[chosen_arm].fit(X_arm, y_arm)
            except Exception as e:
                logger.error(f"Error training model for arm {chosen_arm}: {str(e)}")
    
    def get_stats(self):
        """Return current statistics about the bandit algorithm"""
        return {
            'algorithm': self.name,
            'values': self.values.tolist(),
            'counts': self.counts.tolist(),
            'total_tries': int(np.sum(self.counts))
        }
    
    def get_feature_importance(self, arm):
        """
        Get feature importance for a specific arm.
        
        Args:
            arm (int): Index of the arm
            
        Returns:
            dict: Feature importance scores
        """
        if self.counts[arm] < self.min_samples:
            return {"error": "Not enough data for this arm"}
        
        try:
            # Get coefficients from the logistic regression model
            model = self.models[arm].named_steps['logreg']
            coefficients = model.coef_[0]
            
            # Return coefficients
            return {
                "coefficients": coefficients.tolist(),
                "intercept": model.intercept_[0]
            }
        except Exception as e:
            logger.error(f"Error getting feature importance for arm {arm}: {str(e)}")
            return {"error": str(e)}
    
    def save_state(self, file_path):
        """Save the current state of the bandit algorithm"""
        state = {
            'algorithm': self.name,
            'n_arms': self.n_arms,
            'exploration_factor': self.exploration_factor,
            'counts': self.counts.tolist(),
            'values': self.values.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        # We don't save the models or data here - they're too complex
        # Instead, save just the basic state and metrics
        
        with open(file_path, 'w') as f:
            json.dump(state, f)
        
        # Save models and data separately if needed
        model_dir = os.path.dirname(file_path) + "/models"
        os.makedirs(model_dir, exist_ok=True)
        
        # Save X and y data for each arm
        data_path = os.path.join(model_dir, "bandit_data.npz")
        np.savez(data_path, 
                 X=[np.array(x) if len(x) > 0 else np.array([]) for x in self.X],
                 y=[np.array(y) if len(y) > 0 else np.array([]) for y in self.y])
        
        logger.info(f"Saved {self.name} state to {file_path}")
    
    def load_state(self, file_path):
        """Load a previously saved state"""
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            if state['algorithm'] != self.name:
                logger.warning(f"Loading state for {state['algorithm']} into {self.name} instance")
            
            self.n_arms = state['n_arms']
            self.exploration_factor = state['exploration_factor']
            self.counts = np.array(state['counts'])
            self.values = np.array(state['values'])
            
            # Load models and data separately if they exist
            model_dir = os.path.dirname(file_path) + "/models"
            data_path = os.path.join(model_dir, "bandit_data.npz")
            
            if os.path.exists(data_path):
                data = np.load(data_path, allow_pickle=True)
                self.X = [x for x in data['X']]
                self.y = [y for y in data['y']]
                
                # Retrain models with the loaded data
                for arm in range(self.n_arms):
                    if len(self.X[arm]) >= self.min_samples:
                        X_arm = np.array(self.X[arm])
                        y_arm = np.array(self.y[arm])
                        self.models[arm].fit(X_arm, y_arm)
            
            logger.info(f"Loaded {self.name} state from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
            return False


class ContextualWebDesignBandit:
    """
    Contextual bandit implementation specifically for web design testing
    that considers user context features.
    """
    
    def __init__(self, algorithm_name='linucb', n_designs=3, context_dim=10, state_file=None):
        """
        Initialize the contextual bandit for web design testing.
        
        Args:
            algorithm_name (str): Name of the algorithm to use ('linucb' or 'logistic')
            n_designs (int): Number of design variants
            context_dim (int): Dimension of context features
            state_file (str): Path to a saved state file
        """
        self.n_designs = n_designs
        self.context_dim = context_dim
        
        # Choose algorithm
        if algorithm_name.lower() == 'linucb':
            self.algorithm = LinUCB(n_designs, context_dim)
        elif algorithm_name.lower() == 'logistic':
            self.algorithm = LogisticContextualBandit(n_designs)
        else:
            logger.warning(f"Unknown algorithm '{algorithm_name}', defaulting to LinUCB")
            self.algorithm = LinUCB(n_designs, context_dim)
        
        # Try to load saved state
        if state_file and os.path.exists(state_file):
            self.algorithm.load_state(state_file)
    
    def extract_context_features(self, user_data):
        """
        Extract context features from user data.
        
        Args:
            user_data (dict): Dictionary containing user information
            
        Returns:
            list: Extracted features as a list
        """
        # This is a simple example - in practice, you'd implement more sophisticated feature extraction
        features = []
        
        # Device type (0=desktop, 1=mobile, 2=tablet)
        device_map = {'desktop': 0, 'mobile': 1, 'tablet': 2}
        device = user_data.get('device', 'desktop')
        features.append(device_map.get(device, 0))
        
        # New or returning user (0=new, 1=returning)
        features.append(1 if user_data.get('returning_user', False) else 0)
        
        # Time of day (0-23)
        hour = datetime.now().hour
        features.append(hour)
        
        # Weekend flag (0=weekday, 1=weekend)
        is_weekend = 1 if datetime.now().weekday() >= 5 else 0
        features.append(is_weekend)
        
        # Referrer type
        referrer_map = {'search': 0, 'social': 1, 'direct': 2, 'email': 3, 'other': 4}
        referrer = user_data.get('referrer', 'direct')
        features.append(referrer_map.get(referrer, 4))
        
        # Fill remaining features with zeros if needed
        while len(features) < self.context_dim:
            features.append(0)
        
        # Truncate if too many
        features = features[:self.context_dim]
        
        return features
    
    def select_design(self, user_data=None):
        """
        Select which design to show to a user based on their context.
        
        Args:
            user_data (dict): User context information
            
        Returns:
            int: Selected design index
        """
        # Extract context features from user data
        if user_data is None:
            user_data = {}  # Empty dict if no user data provided
        
        context = self.extract_context_features(user_data)
        
        # Select design using contextual bandit algorithm
        selected_design = self.algorithm.select_arm(context)
        
        logger.info(f"Selected design {selected_design} using {self.algorithm.name} for context {context}")
        return selected_design
    
    def record_conversion(self, design, user_data, converted):
        """
        Record whether the user converted (1) or not (0).
        
        Args:
            design (int): The design that was shown
            user_data (dict): User context information
            converted (bool): Whether the user converted
        """
        reward = 1 if converted else 0
        context = self.extract_context_features(user_data)
        
        self.algorithm.update(design, context, reward)
        
        logger.info(f"Recorded {'conversion' if converted else 'non-conversion'} for design {design} with context {context}")
    
    def get_current_stats(self):
        """
        Get current statistics about designs.
        
        Returns:
            dict: Statistics about the bandit algorithm
        """
        stats = self.algorithm.get_stats()
        stats['design_names'] = [f"Design {i+1}" for i in range(self.n_designs)]
        return stats
    
    def save_state(self, file_path='data/contextual_bandit_state.json'):
        """
        Save current state to file.
        
        Args:
            file_path (str): Path to save the state
        """
        self.algorithm.save_state(file_path)
    
    def generate_report(self, output_file='data/contextual_mab_report.json'):
        """
        Generate a detailed report of the MAB performance.
        
        Args:
            output_file (str): Path to save the report
            
        Returns:
            dict: Generated report
        """
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
            
            # Add feature importance if available
            if hasattr(self.algorithm, 'get_feature_importance'):
                try:
                    importance = self.algorithm.get_feature_importance(i)
                    design_data['feature_importance'] = importance
                except Exception as e:
                    logger.error(f"Error getting feature importance: {str(e)}")
            
            report['designs'].append(design_data)
        
        # Save report to file
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Generated MAB report and saved to {output_file}")
        return report 