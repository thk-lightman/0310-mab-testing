import os
import boto3
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/aws_operations.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('aws_utils')

class AWSManager:
    def __init__(self):
        """Initialize AWS services connections"""
        try:
            # AWS credentials
            aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            aws_region = os.getenv('AWS_REGION', 'us-east-1')
            
            # Initialize AWS session
            self.session = boto3.Session(
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region
            )
            
            # Initialize services
            self.s3 = self.session.client('s3')
            self.dynamodb = self.session.resource('dynamodb')
            self.cloudwatch = self.session.client('cloudwatch')
            
            logger.info("AWS services initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing AWS services: {str(e)}")
            raise
    
    def create_data_bucket(self, bucket_name):
        """Create an S3 bucket for storing application data"""
        try:
            self.s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={
                    'LocationConstraint': self.session.region_name
                }
            )
            logger.info(f"S3 bucket '{bucket_name}' created successfully")
            return True
        except Exception as e:
            logger.error(f"Error creating S3 bucket: {str(e)}")
            return False
    
    def create_dynamodb_table(self, table_name):
        """Create DynamoDB table for tracking MAB test data"""
        try:
            table = self.dynamodb.create_table(
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
            # Wait until the table exists
            table.meta.client.get_waiter('table_exists').wait(TableName=table_name)
            logger.info(f"DynamoDB table '{table_name}' created successfully")
            return True
        except Exception as e:
            logger.error(f"Error creating DynamoDB table: {str(e)}")
            return False
    
    def log_interaction(self, table_name, session_id, design_version, action, success=False):
        """Log user interaction data to DynamoDB"""
        try:
            table = self.dynamodb.Table(table_name)
            timestamp = datetime.now().isoformat()
            
            response = table.put_item(
                Item={
                    'session_id': session_id,
                    'timestamp': timestamp,
                    'design_version': design_version,
                    'action': action,
                    'success': success
                }
            )
            logger.info(f"Interaction logged successfully for session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error logging interaction: {str(e)}")
            return False
    
    def upload_results_to_s3(self, bucket_name, file_path, object_key):
        """Upload MAB results to S3"""
        try:
            self.s3.upload_file(file_path, bucket_name, object_key)
            logger.info(f"Results uploaded successfully to s3://{bucket_name}/{object_key}")
            return True
        except Exception as e:
            logger.error(f"Error uploading results: {str(e)}")
            return False
    
    def download_data_from_s3(self, bucket_name, object_key, local_path):
        """Download data from S3 bucket"""
        try:
            self.s3.download_file(bucket_name, object_key, local_path)
            logger.info(f"Data downloaded successfully from s3://{bucket_name}/{object_key}")
            return True
        except Exception as e:
            logger.error(f"Error downloading data: {str(e)}")
            return False
    
    def export_data_from_dynamodb(self, table_name, output_file):
        """Export all data from DynamoDB table to a JSON file"""
        try:
            table = self.dynamodb.Table(table_name)
            response = table.scan()
            
            with open(output_file, 'w') as f:
                json.dump(response['Items'], f)
                
            logger.info(f"Data exported successfully from DynamoDB table '{table_name}'")
            return True
        except Exception as e:
            logger.error(f"Error exporting data from DynamoDB: {str(e)}")
            return False 