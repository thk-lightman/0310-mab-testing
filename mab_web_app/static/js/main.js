/**
 * Main JavaScript file for MAB Web Testing application
 */

document.addEventListener('DOMContentLoaded', function() {
    console.log('MAB Web Testing application initialized');
    
    // Update countdown timer if it exists
    const countdownElement = document.querySelector('.countdown');
    if (countdownElement) {
        startCountdown(countdownElement);
    }
    
    // Handle tab switching in product tabs if they exist
    const tabElements = document.querySelectorAll('.tab');
    if (tabElements.length > 0) {
        setupTabs();
    }
    
    // Track user interactions
    trackUserInteractions();
});

/**
 * Start a countdown timer
 * @param {HTMLElement} element - The element to display the countdown
 */
function startCountdown(element) {
    // Set the countdown to 24 hours from now
    const endTime = new Date();
    endTime.setHours(endTime.getHours() + 24);
    
    // Update countdown every second
    const countdownInterval = setInterval(function() {
        const now = new Date().getTime();
        const distance = endTime - now;
        
        // Time calculations
        const hours = Math.floor((distance % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
        const minutes = Math.floor((distance % (1000 * 60 * 60)) / (1000 * 60));
        const seconds = Math.floor((distance % (1000 * 60)) / 1000);
        
        // Display the result
        element.textContent = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        
        // If the countdown is finished, clear interval
        if (distance < 0) {
            clearInterval(countdownInterval);
            element.textContent = "00:00:00";
        }
    }, 1000);
}

/**
 * Set up tabbed content functionality
 */
function setupTabs() {
    const tabs = document.querySelectorAll('.tab');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const tabId = tab.getAttribute('data-tab');
            
            // Deactivate all tabs and content
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Activate clicked tab and corresponding content
            tab.classList.add('active');
            document.getElementById(tabId).classList.add('active');
        });
    });
}

/**
 * Track user interactions with the page
 */
function trackUserInteractions() {
    // Track scroll depth
    let maxScrollDepth = 0;
    window.addEventListener('scroll', function() {
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        const scrollHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
        const scrollDepthPercentage = (scrollTop / scrollHeight) * 100;
        
        if (scrollDepthPercentage > maxScrollDepth) {
            maxScrollDepth = scrollDepthPercentage;
            
            // Log significant scroll depth milestones
            if (maxScrollDepth >= 25 && maxScrollDepth < 26) {
                console.log('User reached 25% scroll depth');
            } else if (maxScrollDepth >= 50 && maxScrollDepth < 51) {
                console.log('User reached 50% scroll depth');
            } else if (maxScrollDepth >= 75 && maxScrollDepth < 76) {
                console.log('User reached 75% scroll depth');
            } else if (maxScrollDepth >= 100) {
                console.log('User reached 100% scroll depth');
            }
        }
    });
    
    // Track time spent on page
    const startTime = new Date();
    let timeLogInterval = setInterval(function() {
        const timeSpentInSeconds = Math.floor((new Date() - startTime) / 1000);
        
        // Log time spent every minute
        if (timeSpentInSeconds % 60 === 0 && timeSpentInSeconds > 0) {
            console.log(`User has spent ${timeSpentInSeconds / 60} minutes on the page`);
        }
        
        // Stop tracking after 30 minutes
        if (timeSpentInSeconds > 1800) {
            clearInterval(timeLogInterval);
        }
    }, 1000);
    
    // Track button hover events
    const buttons = document.querySelectorAll('.cta-button');
    buttons.forEach(button => {
        button.addEventListener('mouseenter', function() {
            console.log('User hovered over CTA button');
        });
    });
}

/**
 * Handle form submissions and validate before submit
 * @param {HTMLFormElement} form - The form element
 * @returns {boolean} - Whether form is valid
 */
function validateForm(form) {
    const requiredFields = form.querySelectorAll('[required]');
    let isValid = true;
    
    requiredFields.forEach(field => {
        if (!field.value.trim()) {
            isValid = false;
            
            // Add error styling
            field.classList.add('error');
            
            // Create error message if it doesn't exist
            let errorMsg = field.nextElementSibling;
            if (!errorMsg || !errorMsg.classList.contains('error-message')) {
                errorMsg = document.createElement('div');
                errorMsg.classList.add('error-message');
                errorMsg.textContent = 'This field is required';
                field.parentNode.insertBefore(errorMsg, field.nextSibling);
            }
        } else {
            // Remove error styling
            field.classList.remove('error');
            
            // Remove error message if it exists
            const errorMsg = field.nextElementSibling;
            if (errorMsg && errorMsg.classList.contains('error-message')) {
                errorMsg.remove();
            }
        }
    });
    
    return isValid;
}

// Add form validation to all forms with class 'validate-form'
document.querySelectorAll('.validate-form').forEach(form => {
    form.addEventListener('submit', function(event) {
        if (!validateForm(this)) {
            event.preventDefault();
        }
    });
});

// Handle "Add to Wishlist" button if it exists
const wishlistButtons = document.querySelectorAll('.wishlist-button');
wishlistButtons.forEach(button => {
    button.addEventListener('click', function() {
        const icon = this.querySelector('i');
        
        // Toggle heart icon
        if (icon.classList.contains('far')) {
            icon.classList.remove('far');
            icon.classList.add('fas');
            this.setAttribute('title', 'Remove from Wishlist');
            alert('Added to wishlist!');
        } else {
            icon.classList.remove('fas');
            icon.classList.add('far');
            this.setAttribute('title', 'Add to Wishlist');
            alert('Removed from wishlist!');
        }
    });
}); 