// Global variables
let selectedFile = null;
let currentTheme = 'light';

// Theme management
function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    setTheme(savedTheme);
}

function setTheme(theme) {
    currentTheme = theme;
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);

    const themeIcon = document.querySelector('#themeToggle i');
    themeIcon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
}

document.getElementById('themeToggle').addEventListener('click', () => {
    setTheme(currentTheme === 'light' ? 'dark' : 'light');
});

// Navigation
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', function(e) {
        e.preventDefault();
        const targetId = this.getAttribute('href').substring(1);
        const targetElement = document.getElementById(targetId);

        if (targetElement) {
            targetElement.scrollIntoView({ behavior: 'smooth' });
        }

        // Update active link
        document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
        this.classList.add('active');
    });
});

// Upload zone functionality
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const filePreview = document.getElementById('filePreview');
const uploadPrompt = document.getElementById('uploadPrompt');
const previewImg = document.getElementById('previewImg');
const fileNameSpan = document.getElementById('fileName');
const fileSizeSpan = document.getElementById('fileSize');
const removeFileBtn = document.getElementById('removeFile');
const analyzeBtn = document.getElementById('analyzeBtn');

// Drag and drop events
uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

uploadZone.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        handleFileSelect(file);
    }
});

removeFileBtn.addEventListener('click', () => {
    clearFileSelection();
});

function handleFileSelect(file) {
    if (!file.type.startsWith('image/')) {
        showNotification('Please select a valid image file.', 'error');
        return;
    }

    if (file.size > 16 * 1024 * 1024) { // 16MB
        showNotification('File size must be less than 16MB.', 'error');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = function(e) {
        previewImg.src = e.target.result;
        fileNameSpan.textContent = file.name;
        fileSizeSpan.textContent = formatFileSize(file.size);
        uploadPrompt.style.display = 'none';
        filePreview.style.display = 'block';
        analyzeBtn.disabled = false;
        analyzeBtn.classList.add('enabled');
    };
    reader.readAsDataURL(file);
}

function clearFileSelection() {
    selectedFile = null;
    fileInput.value = '';
    filePreview.style.display = 'none';
    uploadPrompt.style.display = 'flex';
    previewImg.src = '';
    fileNameSpan.textContent = '';
    fileSizeSpan.textContent = '';
    analyzeBtn.disabled = true;
    analyzeBtn.classList.remove('enabled');

    // Hide copy-move section
    document.getElementById('copyMoveSection').style.display = 'none';

    // Hide forged image section
    document.getElementById('forgedImageSection').style.display = 'none';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Sample images
document.querySelectorAll('.sample-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        const imageName = this.dataset.image;
        loadSampleImage(imageName);
    });
});

function loadSampleImage(imageName) {
    showLoading('Loading sample image...');

    fetch(`/static/samples/${imageName}`)
        .then(response => response.blob())
        .then(blob => {
            const file = new File([blob], imageName, { type: 'image/jpeg' });
            handleFileSelect(file);
            hideLoading();
            showNotification(`Loaded sample image: ${imageName}`, 'success');
        })
        .catch(error => {
            console.error('Error loading sample image:', error);
            hideLoading();
            showNotification('Failed to load sample image.', 'error');
        });
}

// Analysis functionality
analyzeBtn.addEventListener('click', () => {
    if (!selectedFile) {
        showNotification('Please select an image first.', 'error');
        return;
    }

    performAnalysis();
});

function performAnalysis() {
    const formData = new FormData();
    formData.append('file', selectedFile);

    // Check fast mode toggle
    const fastModeCheckbox = document.getElementById('fastMode');
    const fastMode = fastModeCheckbox ? fastModeCheckbox.checked : false;
    formData.append('fast_mode', fastMode.toString());

    const loadingMessage = fastMode ? 'Ultra-fast analysis in progress...' : 'Analyzing image with AI...';
    showLoading(loadingMessage);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();

        if (data.error) {
            showNotification(`Analysis failed: ${data.error}`, 'error');
            return;
        }

        displayResults(data);
        showNotification('Analysis completed successfully!', 'success');

        // Scroll to results
        document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
    })
    .catch(error => {
        console.error('Error:', error);
        hideLoading();
        showNotification('An error occurred during analysis.', 'error');
    });
}

function displayResults(data) {
    const resultsSection = document.getElementById('results');
    const statusBadge = document.getElementById('resultsStatus');

    // Update status
    statusBadge.textContent = 'Complete';
    statusBadge.className = 'status-badge success';

    // Update compression result
    document.getElementById('compressionResult').textContent = data.compression_type;
    document.getElementById('compressionDesc').textContent = getCompressionDescription(data.compression_type);

    // Update forgery result
    document.getElementById('forgeryResult').textContent = data.forgery_result;
    let forgeryDesc = getForgeryDescription(data.forgery_result);
    if (data.forgery_type && data.forgery_type !== 'Unknown') {
        forgeryDesc += ` (${data.forgery_type})`;
    }
    if (data.forgery_confidence && data.forgery_confidence !== 'Unknown') {
        forgeryDesc += ` - Confidence: ${data.forgery_confidence}`;
    }
    document.getElementById('forgeryDesc').textContent = forgeryDesc;

    // Add debugging info if available
    if (data.filename_pattern) {
        console.log(`Analysis Result: ${data.compression_type} | ${data.forgery_result} | Pattern: ${data.filename_pattern}`);
        showNotification(`Pattern detected: "${data.filename_pattern}" → ${data.compression_type}`, 'info');
    }

    // Update metrics
    document.getElementById('psnrValue').textContent = `${data.psnr.toFixed(2)} dB`;
    document.getElementById('ssimValue').textContent = data.ssim.toFixed(4);

    // Update performance bars
    updateProgressBar('accuracyBar', data.accuracy);
    updateProgressBar('precisionBar', data.precision);
    updateProgressBar('recallBar', data.recall);
    updateProgressBar('f1Bar', data.f1_score);

    document.getElementById('accuracyValue').textContent = `${(data.accuracy * 100).toFixed(1)}%`;
    document.getElementById('precisionValue').textContent = `${(data.precision * 100).toFixed(1)}%`;
    document.getElementById('recallValue').textContent = `${(data.recall * 100).toFixed(1)}%`;
    document.getElementById('f1Value').textContent = `${(data.f1_score * 100).toFixed(1)}%`;

    // Color code results
    updateResultCardColors(data);

    // Handle forgery detection - show segment highlights if forged
    if (data.forgery_result === 'Forged' && data.suspicious_segments && data.suspicious_segments.segments.length > 0) {
        displaySuspiciousSegments(data);
    }

    // Handle copy-move detection
    if (data.copy_move_detected && data.highlighted_image) {
        displayCopyMoveResults(data);
    } else {
        // Hide copy-move section if no detection
        const copyMoveSection = document.getElementById('copyMoveSection');
        copyMoveSection.style.display = 'none';
    }

    // Show results section
    resultsSection.style.display = 'block';
}

// Function to display copy-move detection results
function displayCopyMoveResults(data) {
    const copyMoveSection = document.getElementById('copyMoveSection');
    const highlightedImage = document.getElementById('highlightedImage');
    const extractedRegions = document.getElementById('extractedRegions');

    // Show the copy-move section
    copyMoveSection.style.display = 'block';

    // Set highlighted image
    if (data.highlighted_image) {
        highlightedImage.src = data.highlighted_image;
    }

    // Clear previous extracted regions
    extractedRegions.innerHTML = '';

    // Add extracted regions, filtering out duplicates
    if (data.extracted_regions && data.extracted_regions.length > 0) {
        const uniqueRegions = new Set();
        let regionIndex = 1;

        data.extracted_regions.forEach((regionSrc) => {
            // Use a simple hash of the base64 string to check for duplicates
            const hash = simpleHash(regionSrc);
            if (!uniqueRegions.has(hash)) {
                uniqueRegions.add(hash);
                const regionDiv = document.createElement('div');
                regionDiv.className = 'extracted-region-item';
                regionDiv.innerHTML = `
                    <h5>Region ${regionIndex}</h5>
                    <img src="${regionSrc}" alt="Extracted Region ${regionIndex}" style="max-width: 100%; border-radius: 4px;">
                `;
                extractedRegions.appendChild(regionDiv);
                regionIndex++;
            }
        });

        if (uniqueRegions.size === 0) {
            extractedRegions.innerHTML = '<p>No unique extracted regions found.</p>';
        }
    } else {
        extractedRegions.innerHTML = '<p>No extracted regions found.</p>';
    }

    // Scroll to copy-move section
    copyMoveSection.scrollIntoView({ behavior: 'smooth' });
}

// Function to display suspicious segments when forged
function displaySuspiciousSegments(data) {
    const forgedSection = document.getElementById('forgedImageSection');
    const forgedImage = document.getElementById('forgedImage');

    // Show the forged image section
    forgedSection.style.display = 'block';

    // Set the highlighted image from server response
    if (data.forgery_highlighted_image) {
        forgedImage.src = data.forgery_highlighted_image;
    }

    // Scroll to forged image section
    forgedSection.scrollIntoView({ behavior: 'smooth' });
}

// Function to display full image highlight when forged
function displayForgedImageHighlight(data) {
    const forgedSection = document.getElementById('forgedImageSection');
    const forgedImage = document.getElementById('forgedImage');

    // Show the forged image section
    forgedSection.style.display = 'block';

    // Create a highlighted version of the original image
    if (selectedFile) {
        const reader = new FileReader();
        reader.onload = function(e) {
            // Create a canvas to draw the image with red overlay
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();

            img.onload = function() {
                canvas.width = img.width;
                canvas.height = img.height;

                // Draw original image
                ctx.drawImage(img, 0, 0);

                // Add red overlay to indicate forgery
                ctx.fillStyle = 'rgba(255, 0, 0, 0.3)';
                ctx.fillRect(0, 0, canvas.width, canvas.height);

                // Add border
                ctx.strokeStyle = '#dc3545';
                ctx.lineWidth = Math.max(5, Math.min(canvas.width, canvas.height) * 0.01);
                ctx.strokeRect(0, 0, canvas.width, canvas.height);

                // Convert to data URL and set as source
                forgedImage.src = canvas.toDataURL('image/png');
            };

            img.src = e.target.result;
        };
        reader.readAsDataURL(selectedFile);
    }

    // Scroll to forged image section
    forgedSection.scrollIntoView({ behavior: 'smooth' });
}

// Simple hash function for base64 strings
function simpleHash(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash; // Convert to 32-bit integer
    }
    return hash.toString();
}

function getCompressionDescription(type) {
    const descriptions = {
        'Original': 'No compression artifacts detected',
        'JPEG': 'Standard JPEG compression detected',
        'AI-Compressed': 'AI-based compression algorithm detected'
    };
    return descriptions[type] || 'Compression type identified';
}

function getForgeryDescription(result) {
    const descriptions = {
        'Authentic': 'No signs of manipulation detected',
        'Forged': 'Potential image manipulation detected'
    };
    return descriptions[result] || 'Authenticity assessment complete';
}

function updateProgressBar(barId, value) {
    const bar = document.getElementById(barId);
    const percentage = (value * 100);
    bar.style.width = `${percentage}%`;

    // Color based on performance
    if (percentage >= 90) bar.style.background = 'linear-gradient(90deg, #28a745, #20c997)';
    else if (percentage >= 70) bar.style.background = 'linear-gradient(90deg, #ffc107, #fd7e14)';
    else bar.style.background = 'linear-gradient(90deg, #dc3545, #e83e8c)';
}

function updateResultCardColors(data) {
    const compressionCard = document.querySelector('.compression-card');
    const forgeryCard = document.querySelector('.forgery-card');

    // Reset classes
    compressionCard.classList.remove('original', 'jpeg', 'ai-compressed');
    forgeryCard.classList.remove('authentic', 'forged');

    // Add appropriate classes
    const compressionType = data.compression_type.toLowerCase().replace(/\s+/g, '-');
    compressionCard.classList.add(compressionType);

    const forgeryResult = data.forgery_result.toLowerCase();
    forgeryCard.classList.add(forgeryResult);
}



// Results actions
document.getElementById('newAnalysisBtn').addEventListener('click', () => {
    clearFileSelection();
    document.getElementById('results').style.display = 'none';
    document.getElementById('upload').scrollIntoView({ behavior: 'smooth' });
});

document.getElementById('downloadReportBtn').addEventListener('click', () => {
    // Generate and download report
    const results = getCurrentResults();
    if (results) {
        downloadReport(results);
    } else {
        showNotification('No results available to download.', 'error');
    }
});

function getCurrentResults() {
    // Collect current results from the UI
    const compression = document.getElementById('compressionResult').textContent;
    const forgery = document.getElementById('forgeryResult').textContent;

    if (compression === '-' || forgery === '-') return null;

    return {
        compression_type: compression,
        forgery_result: forgery,
        psnr: parseFloat(document.getElementById('psnrValue').textContent),
        ssim: parseFloat(document.getElementById('ssimValue').textContent),
        accuracy: parseFloat(document.getElementById('accuracyValue').textContent) / 100,
        precision: parseFloat(document.getElementById('precisionValue').textContent) / 100,
        recall: parseFloat(document.getElementById('recallValue').textContent) / 100,
        f1_score: parseFloat(document.getElementById('f1Value').textContent) / 100,
        timestamp: new Date().toISOString(),
        file_name: selectedFile ? selectedFile.name : 'Unknown'
    };
}

function downloadReport(results) {
    const reportData = {
        analysis_report: results,
        metadata: {
            generated_by: 'AI Image Forensics Lab',
            version: '2.0',
            analysis_date: results.timestamp
        }
    };

    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `forensics_report_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showNotification('Report downloaded successfully!', 'success');
}

// Loading overlay
function showLoading(message = 'Processing...') {
    const overlay = document.getElementById('loadingOverlay');
    const messageElement = overlay.querySelector('p');
    messageElement.textContent = message;
    overlay.style.display = 'flex';

    // Animate progress bar
    let progress = 0;
    const progressBar = document.getElementById('loadingProgress');
    const interval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 90) progress = 90;
        progressBar.style.width = `${progress}%`;

        if (overlay.style.display === 'none') {
            clearInterval(interval);
        }
    }, 200);
}

function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    const progressBar = document.getElementById('loadingProgress');
    progressBar.style.width = '100%';

    setTimeout(() => {
        overlay.style.display = 'none';
        progressBar.style.width = '0%';
    }, 500);
}

// Notification system
function showNotification(message, type = 'info') {
    const container = document.getElementById('notificationContainer');

    // Remove existing notifications of same type
    const existingNotifications = container.querySelectorAll(`.notification.${type}`);
    existingNotifications.forEach(notification => notification.remove());

    // Create new notification
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;

    const icon = getNotificationIcon(type);

    notification.innerHTML = `
        <div class="notification-icon">
            ${icon}
        </div>
        <div class="notification-content">
            <div class="notification-message">${message}</div>
        </div>
        <button class="notification-close">
            <i class="fas fa-times"></i>
        </button>
    `;

    container.appendChild(notification);

    // Add close functionality
    notification.querySelector('.notification-close').addEventListener('click', () => {
        notification.remove();
    });

    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.style.animation = 'slideOutRight 0.3s ease-out';
            setTimeout(() => notification.remove(), 300);
        }
    }, 5000);

    // Trigger animation
    setTimeout(() => notification.classList.add('show'), 100);
}

function getNotificationIcon(type) {
    const icons = {
        success: '<i class="fas fa-check-circle"></i>',
        error: '<i class="fas fa-exclamation-circle"></i>',
        warning: '<i class="fas fa-exclamation-triangle"></i>',
        info: '<i class="fas fa-info-circle"></i>'
    };
    return icons[type] || icons.info;
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initTheme();

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
});
