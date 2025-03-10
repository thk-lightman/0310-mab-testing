#!/usr/bin/env python3
"""
Script to set up necessary AWS resources for MAB Web Testing application.
This creates:
1. S3 bucket for storing results
2. DynamoDB table for storing interaction data
"""

import os
import sys
import argparse
import logging
import boto3
from dotenv import load_dotenv
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/aws_setup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('aws_setup')

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Set up AWS resources for MAB Web Testing')
    parser.add_argument('-b', '--bucket', type=str, help='S3 bucket name')
    parser.add_argument('-t', '--table', type=str, help='DynamoDB table name')
    parser.add_argument('-r', '--region', type=str, help='AWS region')
    parser.add_argument('-f', '--force', action='store_true', help='Force creation of resources even if they exist')
    
    return parser.parse_args()

def init_aws_clients(region=None):
    """Initialize AWS clients"""
    # Load environment variables
    load_dotenv()
    
    # Get AWS credentials from environment variables
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = region or os.getenv('AWS_REGION', 'us-east-1')
    
    if not aws_access_key or not aws_secret_key:
        logger.error("AWS credentials not found. Make sure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set.")
        sys.exit(1)
    
    # Initialize session
    session = boto3.Session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )
    
    # Initialize services
    s3 = session.client('s3')
    dynamodb = session.resource('dynamodb')
    
    return s3, dynamodb, aws_region

def create_s3_bucket(s3_client, bucket_name, region):
    """Create S3 bucket for storing results"""
    try:
        # Check if bucket exists
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"S3 bucket '{bucket_name}' already exists.")
            return True
        except:
            pass  # Bucket doesn't exist, so create it
        
        # Create bucket
        if region == 'us-east-1':
            # Special case for us-east-1 region
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': region}
            )
        
        # Enable versioning
        s3_client.put_bucket_versioning(
            Bucket=bucket_name,
            VersioningConfiguration={'Status': 'Enabled'}
        )
        
        # Add bucket policy for private access
        policy = {
            'Version': '2012-10-17',
            'Statement': [{
                'Sid': 'PrivateAccess',
                'Effect': 'Deny',
                'Principal': '*',
                'Action': 's3:GetObject',
                'Resource': f'arn:aws:s3:::{bucket_name}/*',
                'Condition': {
                    'StringNotEquals': {
                        'aws:PrincipalAccount': os.getenv('AWS_ACCOUNT_ID', '')
                    }
                }
            }]
        }
        
        # Only set policy if AWS_ACCOUNT_ID is available
        if os.getenv('AWS_ACCOUNT_ID'):
            s3_client.put_bucket_policy(
                Bucket=bucket_name,
                Policy=json.dumps(policy)
            )
        
        logger.info(f"S3 bucket '{bucket_name}' created successfully in region '{region}'.")
        return True
        
    except Exception as e:
        logger.error(f"Error creating S3 bucket: {str(e)}")
        return False

def create_dynamodb_table(dynamodb_resource, table_name):
    """Create DynamoDB table for storing interaction data"""
    try:
        # Check if table exists
        existing_tables = dynamodb_resource.meta.client.list_tables()['TableNames']
        if table_name in existing_tables:
            logger.info(f"DynamoDB table '{table_name}' already exists.")
            return True
        
        # Create table
        table = dynamodb_resource.create_table(
            TableName=table_name,
            KeySchema=[
                {'AttributeName': 'session_id', 'KeyType': 'HASH'},
                {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}
            ],
            AttributeDefinitions=[
                {'AttributeName': 'session_id', 'AttributeType': 'S'},
                {'AttributeName': 'timestamp', 'AttributeType': 'S'},
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': 5,
                'WriteCapacityUnits': 5
            }
        )
        
        # Wait until table exists
        table.meta.client.get_waiter('table_exists').wait(TableName=table_name)
        logger.info(f"DynamoDB table '{table_name}' created successfully.")
        return True
        
    except Exception as e:
        logger.error(f"Error creating DynamoDB table: {str(e)}")
        return False

def create_config_file(bucket_name, table_name, region):
    """Create config file with AWS resource information"""
    try:
        config = {
            'aws_resources': {
                's3_bucket': bucket_name,
                'dynamodb_table': table_name,
                'region': region
            },
            'created_at': datetime.now().isoformat()
        }
        
        with open('data/aws_config.json', 'w') as f:
            json.dump(config, f, indent=4)
        
        logger.info("AWS configuration saved to 'data/aws_config.json'.")
        return True
        
    except Exception as e:
        logger.error(f"Error creating config file: {str(e)}")
        return False

def main():
    """Main function"""
    # Ensure directories exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Parse arguments
    args = parse_args()
    
    # Initialize AWS clients
    s3_client, dynamodb_resource, region = init_aws_clients(args.region)
    
    # Set resource names (using arguments or defaults)
    bucket_name = args.bucket or os.getenv('S3_BUCKET', 'mab-web-testing-results')
    table_name = args.table or os.getenv('DYNAMODB_TABLE', 'mab_interactions')
    
    # Create S3 bucket
    bucket_created = create_s3_bucket(s3_client, bucket_name, region)
    
    # Create DynamoDB table
    table_created = create_dynamodb_table(dynamodb_resource, table_name)
    
    # Create config file
    if bucket_created and table_created:
        create_config_file(bucket_name, table_name, region)
        logger.info("AWS resources have been successfully set up!")
    else:
        logger.warning("Some AWS resources could not be created. See log for details.")
        sys.exit(1)

if __name__ == '__main__':
    main() 