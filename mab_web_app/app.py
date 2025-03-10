import os
import json
import logging
import uuid
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from dotenv import load_dotenv

# Import custom modules
from mab import WebDesignBandit
from aws_utils import AWSManager
from visualize import MABVisualizer

# Load environment variables
load_dotenv()

# Configure logging
os.makedirs('logs', exist_ok=True)
os.makedirs('data', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('mab_app')

# Initialize Flask app
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')

# Set secret key for session management
app.secret_key = os.getenv('SECRET_KEY', os.urandom(24).hex())

# Initialize MAB algorithm
mab_algorithm = os.getenv('MAB_ALGORITHM', 'thompson')
bandit = WebDesignBandit(
    algorithm_name=mab_algorithm,
    n_designs=3,
    state_file='data/bandit_state.json'
)

# Initialize AWS manager if AWS credentials are provided
aws_enabled = all([
    os.getenv('AWS_ACCESS_KEY_ID'),
    os.getenv('AWS_SECRET_ACCESS_KEY')
])

aws_manager = None
if aws_enabled:
    try:
        aws_manager = AWSManager()
        logger.info("AWS integration enabled")
    except Exception as e:
        logger.error(f"Failed to initialize AWS manager: {str(e)}")
        aws_enabled = False

# Initialize visualizer
visualizer = MABVisualizer(data_dir='data', output_dir='static/images')

# DynamoDB table name for tracking user interactions
DYNAMODB_TABLE = os.getenv('DYNAMODB_TABLE', 'mab_interactions')

# S3 bucket for storing results
S3_BUCKET = os.getenv('S3_BUCKET', 'mab-web-testing-results')

# Dictionary to store session data locally if AWS is not available
local_sessions = {}

@app.before_request
def before_request():
    """Set up user session if it doesn't exist"""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
        logger.info(f"New session created: {session['user_id']}")


@app.route('/')
def index():
    """Main entry point - redirect to a design based on MAB selection"""
    # Select design using MAB
    selected_design = bandit.select_design()
    
    # Store design in session
    session['design'] = selected_design
    
    # Log interaction
    log_interaction(session['user_id'], selected_design, 'impression', success=False)
    
    # Redirect to appropriate design
    return redirect(url_for(f'design{selected_design + 1}'))


@app.route('/design1')
def design1():
    """Design version 1"""
    return render_template('design1.html')


@app.route('/design2')
def design2():
    """Design version 2"""
    return render_template('design2.html')


@app.route('/design3')
def design3():
    """Design version 3"""
    return render_template('design3.html')


@app.route('/convert', methods=['POST'])
def convert():
    """Handle conversion actions (e.g., form submission, click, etc.)"""
    if 'design' not in session:
        logger.warning("Conversion attempt without design in session")
        return redirect(url_for('index'))
    
    # Record conversion for the design in session
    design = session['design']
    bandit.record_conversion(design, True)
    
    # Log interaction
    log_interaction(session['user_id'], design, 'conversion', success=True)
    
    # Save MAB state
    bandit.save_state()
    
    # Generate report
    bandit.generate_report()
    
    # Redirect to thank you page
    return redirect(url_for('thank_you'))


@app.route('/thank-you')
def thank_you():
    """Thank you page after conversion"""
    return render_template('thank_you.html')


@app.route('/admin')
def admin():
    """Admin dashboard to view MAB performance"""
    # Get current MAB stats
    stats = bandit.get_current_stats()
    
    # Generate visualizations
    visualizer.generate_all_visualizations()
    
    # Get image paths for visualizations
    image_dir = os.path.join(app.static_folder, 'images')
    images = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    
    # Sort images by creation time (newest first)
    images.sort(key=lambda x: os.path.getctime(os.path.join(image_dir, x)), reverse=True)
    
    # Group images by type
    image_groups = {}
    for img in images:
        img_type = img.split('_')[0]
        if img_type not in image_groups:
            image_groups[img_type] = []
        if len(image_groups[img_type]) < 1:  # Only keep the newest of each type
            image_groups[img_type].append(img)
    
    # Flatten the dictionary
    latest_images = []
    for group in image_groups.values():
        latest_images.extend(group)
    
    return render_template('admin.html', stats=stats, images=latest_images)


@app.route('/api/stats')
def get_stats():
    """API endpoint to get current MAB stats"""
    stats = bandit.get_current_stats()
    return jsonify(stats)


@app.route('/api/report')
def get_report():
    """API endpoint to get the latest MAB report"""
    try:
        with open('data/mab_report.json', 'r') as f:
            report = json.load(f)
        return jsonify(report)
    except FileNotFoundError:
        return jsonify({"error": "Report not found"}), 404


@app.route('/api/visualize')
def trigger_visualization():
    """API endpoint to trigger visualization generation"""
    images = visualizer.generate_all_visualizations()
    return jsonify({
        "success": True,
        "images": [os.path.basename(img) for img in images if img]
    })


def log_interaction(user_id, design, action, success=False):
    """Log user interaction to AWS DynamoDB or local storage"""
    timestamp = datetime.now().isoformat()
    
    # Create interaction data
    interaction = {
        'session_id': user_id,
        'timestamp': timestamp,
        'design_version': design,
        'action': action,
        'success': success
    }
    
    # Log to DynamoDB if AWS is enabled
    if aws_enabled and aws_manager:
        try:
            aws_manager.log_interaction(DYNAMODB_TABLE, user_id, design, action, success)
        except Exception as e:
            logger.error(f"Failed to log to DynamoDB: {str(e)}")
            # Fall back to local storage
            store_interaction_locally(interaction)
    else:
        # Store locally
        store_interaction_locally(interaction)


def store_interaction_locally(interaction):
    """Store interaction data locally if AWS is not available"""
    # Append to local sessions dictionary
    if 'interactions' not in local_sessions:
        local_sessions['interactions'] = []
    
    local_sessions['interactions'].append(interaction)
    
    # Periodically save to file
    if len(local_sessions['interactions']) % 10 == 0:  # Save every 10 interactions
        save_local_interactions()


def save_local_interactions():
    """Save local interactions to file"""
    try:
        with open('data/interactions.json', 'w') as f:
            json.dump(local_sessions['interactions'], f)
        logger.info(f"Saved {len(local_sessions['interactions'])} interactions to local file")
    except Exception as e:
        logger.error(f"Failed to save local interactions: {str(e)}")


@app.route('/debug')
def debug():
    """Debug endpoint to view session information"""
    debug_info = {
        'session_id': session.get('user_id', 'Not set'),
        'design': session.get('design', 'Not set'),
        'bandit_algorithm': bandit.algorithm.name,
        'aws_enabled': aws_enabled,
    }
    return jsonify(debug_info)


@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    logger.error(f"Server error: {str(e)}")
    return render_template('500.html'), 500


if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    
    # Generate initial report if it doesn't exist
    if not os.path.exists('data/mab_report.json'):
        bandit.generate_report()
    
    # Check if templates exist, if not create them
    for i in range(1, 4):
        template_file = f'templates/design{i}.html'
        if not os.path.exists(template_file):
            logger.warning(f"Template {template_file} not found, creating default template")
            os.makedirs('templates', exist_ok=True)
            with open(template_file, 'w') as f:
                f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Design {i} - MAB Testing</title>
    <link rel="stylesheet" href="/static/css/design{i}.css">
</head>
<body class="design{i}">
    <header>
        <h1>Welcome to Design {i}</h1>
    </header>
    <main>
        <div class="content">
            <h2>Product Showcase</h2>
            <p>This is Design {i} of our website. We're testing different layouts to see which one performs best.</p>
            
            <div class="product">
                <img src="/static/images/product.jpg" alt="Product Image">
                <h3>Amazing Product</h3>
                <p>This product will change your life. Buy it now for amazing results!</p>
                <form action="/convert" method="post">
                    <button type="submit" class="cta-button">Buy Now!</button>
                </form>
            </div>
        </div>
    </main>
    <footer>
        <p>&copy; 2023 MAB Testing Company</p>
    </footer>
    <script src="/static/js/main.js"></script>
</body>
</html>""")
    
    # Start the Flask app
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 