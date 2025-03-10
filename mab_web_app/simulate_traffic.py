#!/usr/bin/env python3
"""
Script to simulate website traffic and conversions for MAB testing.
This script generates random user sessions and simulates interactions 
with the different website designs to generate data for MAB analysis.
"""

import os
import sys
import argparse
import logging
import json
import random
import time
import uuid
from datetime import datetime, timedelta
import numpy as np
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import MAB utilities
from mab_web_app.mab import WebDesignBandit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/simulation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('simulation')

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Simulate website traffic for MAB testing')
    parser.add_argument('-u', '--users', type=int, default=1000, help='Number of users to simulate')
    parser.add_argument('-d', '--days', type=int, default=7, help='Number of days to simulate')
    parser.add_argument('-a', '--algorithm', type=str, default='thompson', help='MAB algorithm to use')
    parser.add_argument('-s', '--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('-o', '--output', type=str, default='data/simulation_results.json', help='Output file')
    
    return parser.parse_args()

def generate_session_id():
    """Generate a unique session ID"""
    return str(uuid.uuid4())

def simulate_conversion(design_index, conversion_rates):
    """Simulate whether a user converts based on the design's conversion rate"""
    return random.random() < conversion_rates[design_index]

def get_timestamp(day, hour_range=(8, 22)):
    """Generate a random timestamp for the given day within the hour range"""
    hour = random.randint(hour_range[0], hour_range[1])
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    
    return day.replace(hour=hour, minute=minute, second=second).isoformat()

def main():
    """Main simulation function"""
    # Ensure directories exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Parse arguments
    args = parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Set true conversion rates for each design (unknown to the algorithm)
    # These values represent the "true" performance of each design that we want to discover
    true_conversion_rates = [0.05, 0.08, 0.03]  # Example: Design 2 performs best
    
    logger.info(f"Starting simulation with {args.users} users over {args.days} days")
    logger.info(f"True conversion rates: {true_conversion_rates}")
    
    # Initialize MAB
    bandit = WebDesignBandit(
        algorithm_name=args.algorithm,
        n_designs=len(true_conversion_rates),
        state_file=None  # No state file, start fresh
    )
    
    # Initialize simulation variables
    start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    interactions = []
    designs_shown = [0, 0, 0]  # Count of times each design is shown
    conversions = [0, 0, 0]  # Count of conversions for each design
    
    # Simulate user sessions
    for user_num in range(args.users):
        # Generate session ID
        session_id = generate_session_id()
        
        # Randomly select a day for this session
        day_offset = random.randint(0, args.days - 1)
        session_day = start_date + timedelta(days=day_offset)
        
        # Select design using MAB
        selected_design = bandit.select_design()
        designs_shown[selected_design] += 1
        
        # Log impression
        impression_timestamp = get_timestamp(session_day)
        interactions.append({
            'session_id': session_id,
            'timestamp': impression_timestamp,
            'design_version': selected_design,
            'action': 'impression',
            'success': False
        })
        
        # Simulate conversion based on true conversion rate
        converted = simulate_conversion(selected_design, true_conversion_rates)
        
        # If converted, log conversion
        if converted:
            conversions[selected_design] += 1
            # Conversion happens 1-5 minutes after impression
            minutes_later = random.randint(1, 5)
            conversion_timestamp = (datetime.fromisoformat(impression_timestamp) + 
                                 timedelta(minutes=minutes_later)).isoformat()
            
            interactions.append({
                'session_id': session_id,
                'timestamp': conversion_timestamp,
                'design_version': selected_design,
                'action': 'conversion',
                'success': True
            })
            
            # Update MAB with conversion
            bandit.record_conversion(selected_design, True)
        else:
            # If not converted, still update MAB (no conversion)
            bandit.record_conversion(selected_design, False)
        
        # Provide progress updates
        if (user_num + 1) % 100 == 0 or user_num == args.users - 1:
            completion = (user_num + 1) / args.users * 100
            logger.info(f"Processed {user_num + 1}/{args.users} users ({completion:.1f}%)")
    
    # Save interactions to file
    with open('data/interactions.json', 'w') as f:
        json.dump(interactions, f, indent=2)
    
    # Save MAB state
    bandit.save_state('data/bandit_state.json')
    
    # Generate report
    report = bandit.generate_report('data/mab_report.json')
    
    # Calculate actual conversion rates from simulation
    actual_rates = [
        conversions[i] / designs_shown[i] if designs_shown[i] > 0 else 0
        for i in range(len(true_conversion_rates))
    ]
    
    # Generate summary
    summary = {
        'simulation_parameters': {
            'users': args.users,
            'days': args.days,
            'algorithm': args.algorithm,
            'seed': args.seed
        },
        'true_conversion_rates': true_conversion_rates,
        'actual_conversion_rates': actual_rates,
        'impressions_per_design': designs_shown,
        'conversions_per_design': conversions,
        'timestamp': datetime.now().isoformat(),
        'mab_values': bandit.algorithm.values.tolist()
    }
    
    # Save summary to file
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    logger.info("\nSimulation Summary:")
    logger.info(f"Algorithm: {args.algorithm}")
    logger.info(f"True conversion rates: {true_conversion_rates}")
    logger.info(f"MAB estimated conversion rates: {bandit.algorithm.values.tolist()}")
    logger.info(f"Actual conversion rates: {actual_rates}")
    logger.info(f"Impressions per design: {designs_shown}")
    logger.info(f"Conversions per design: {conversions}")
    logger.info(f"Total users: {args.users}")
    logger.info(f"Total conversions: {sum(conversions)}")
    logger.info(f"Overall conversion rate: {sum(conversions) / args.users:.2%}")
    logger.info(f"Results saved to: {args.output}")
    
    # Return the best design according to the MAB algorithm
    best_design = np.argmax(bandit.algorithm.values)
    logger.info(f"Best design according to MAB: Design {best_design + 1}")
    
    # Check if MAB found the true best design
    true_best = np.argmax(true_conversion_rates)
    if best_design == true_best:
        logger.info("✓ MAB correctly identified the best design!")
    else:
        logger.info(f"✗ MAB did not identify the true best design (Design {true_best + 1})")
    
    return summary

if __name__ == '__main__':
    main() 