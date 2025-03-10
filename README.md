# MAB Web Design Testing

This project implements a Multi-Armed Bandit (MAB) testing framework for evaluating the performance of 3 different web page designs. The application is designed to be deployed on AWS.

## Project Structure

- `app.py`: Main Flask application
- `mab.py`: Multi-Armed Bandit algorithm implementation
- `aws_utils.py`: AWS integration utilities
- `templates/`: HTML templates for different designs
- `static/`: Static files (CSS, JS, images)
- `data/`: Folder for storing collected data
- `logs/`: Application logs
- `visualize.py`: Results visualization script

## Setup Instructions

1. Ensure Python 3.12+ is installed
2. Set up a virtual environment:
   ```
   python -m venv .mab_simulator
   source .mab_simulator/bin/activate  # On Windows: .mab_simulator\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Configure AWS credentials (see AWS Setup section)
5. Run the application:
   ```
   python app.py
   ```

## AWS Setup

1. Create an AWS account if you don't have one
2. Set up AWS credentials in `.env` file or environment variables:
   ```
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_REGION=your_preferred_region
   ```
3. Deploy the application using instructions in `aws_deployment.md`

## Usage

- Access the application at `http://localhost:5000` when running locally
- Monitor MAB performance in the admin dashboard
- Run `python visualize.py` to generate visualization of test results 