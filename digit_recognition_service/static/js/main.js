/**
 * Optical Digit Verification System (ODVS)
 * Client-side Controller & AJAX Engine
 * Strict Rule: Zero emojis/icons, formal document processing behavior
 */

let currentTab = 'upload';
let selectedFile = null;

document.addEventListener('DOMContentLoaded', () => {
    checkSystemHealth();
    initDropZone();
});

// Check Server and Model Health Status
async function checkSystemHealth() {
    const statusElem = document.getElementById('statusIndicator');
    try {
        const response = await fetch('/health');
        if (response.ok) {
            const data = await response.json();
            if (data.model_loaded) {
                statusElem.textContent = 'ONLINE [MODEL READY]';
                statusElem.className = 'telemetry-value status-online';
            } else {
                statusElem.textContent = 'STANDBY [MODEL UNLOADED]';
                statusElem.style.color = '#f59e0b';
            }
        } else {
            statusElem.textContent = 'ERROR [HTTP ' + response.status + ']';
            statusElem.style.color = '#ef4444';
        }
    } catch (err) {
        statusElem.textContent = 'DISCONNECTED';
        statusElem.style.color = '#ef4444';
    }
}

// Tab Switching
function switchTab(tab) {
    currentTab = tab;
    const tabUploadBtn = document.getElementById('tabUploadBtn');
    const tabUrlBtn = document.getElementById('tabUrlBtn');
    const paneUpload = document.getElementById('paneUpload');
    const paneUrl = document.getElementById('paneUrl');

    hideAlert();

    if (tab === 'upload') {
        tabUploadBtn.classList.add('active');
        tabUrlBtn.classList.remove('active');
        paneUpload.style.display = 'block';
        paneUrl.style.display = 'none';
    } else {
        tabUrlBtn.classList.add('active');
        tabUploadBtn.classList.remove('active');
        paneUrl.style.display = 'block';
        paneUpload.style.display = 'none';
    }
}

// Drag & Drop Configuration
function initDropZone() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');

    dropZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('drag-over');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('drag-over');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        if (dt && dt.files && dt.files.length > 0) {
            handleFileSelection(dt.files[0]);
        }
    });
}

function handleFileSelection(file) {
    hideAlert();
    const allowed = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp'];
    const ext = file.name.split('.').pop().toLowerCase();
    const allowedExts = ['png', 'jpg', 'jpeg', 'webp'];

    if (!allowed.includes(file.type) && !allowedExts.includes(ext)) {
        showAlert(
            'Invalid File Type',
            'Selected file is not an allowed image format. Only PNG, JPG, JPEG, and WEBP are supported.',
            'نوع الملف غير مدعوم. يدعم النظام ملفات الصور من النوع PNG, JPG, JPEG, WEBP فقط.'
        );
        clearSelectedFile();
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        showAlert(
            'File Exceeds Size Limit',
            'File size exceeds the 10 MB maximum limit.',
            'حجم الملف يتجاوز الحد الأقصى المسموح به (10 ميغابايت).'
        );
        clearSelectedFile();
        return;
    }

    selectedFile = file;
    document.getElementById('selectedFileName').textContent = file.name;
    document.getElementById('selectedFileSize').textContent = `(${(file.size / 1024).toFixed(1)} KB)`;
    document.getElementById('selectedFileDisplay').style.display = 'flex';
}

function clearSelectedFile() {
    selectedFile = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('selectedFileDisplay').style.display = 'none';
}

// Verification Execution
async function submitVerification() {
    hideAlert();

    const idleState = document.getElementById('idleState');
    const loadingState = document.getElementById('loadingState');
    const resultsContainer = document.getElementById('resultsContainer');
    const submitBtn = document.getElementById('submitBtn');

    let requestBody = null;
    let requestHeaders = {};

    if (currentTab === 'upload') {
        if (!selectedFile) {
            showAlert(
                'Missing Input Document',
                'Please select or drag-and-drop an image file before executing verification.',
                'يرجى اختيار أو إسقاط ملف صورة قبل تنفيذ عملية الفحص.'
            );
            return;
        }
        const formData = new FormData();
        formData.append('file', selectedFile);
        requestBody = formData;
    } else {
        const urlInput = document.getElementById('urlInput');
        const urlValue = urlInput.value.trim();

        if (!urlValue) {
            showAlert(
                'Missing Image URL',
                'Please provide a valid HTTP or HTTPS image URL.',
                'يرجى إدخال رابط صورة صالح يبدأ بـ http:// أو https://'
            );
            return;
        }

        if (!urlValue.startsWith('http://') && !urlValue.startsWith('https://')) {
            showAlert(
                'Invalid URL Protocol',
                'Image URL must begin with http:// or https://',
                'يجب أن يبدأ رابط الصورة ببروتوكول صالح (http:// أو https://)'
            );
            return;
        }

        requestBody = JSON.stringify({ image_url: urlValue });
        requestHeaders['Content-Type'] = 'application/json';
    }

    // Set UI to loading state
    idleState.style.display = 'none';
    resultsContainer.style.display = 'none';
    loadingState.style.display = 'block';
    submitBtn.disabled = true;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: requestHeaders,
            body: requestBody
        });

        const data = await response.json();

        loadingState.style.display = 'none';
        submitBtn.disabled = false;

        if (!response.ok || !data.success) {
            idleState.style.display = 'block';
            showAlert(
                'Verification Processing Failure',
                data.error || 'Server error encountered during image processing or inference.',
                data.error_ar || 'حدث خطأ في الخادم أثناء معالجة الصورة أو التنبؤ.'
            );
            return;
        }

        // Render Results
        renderResults(data);

    } catch (err) {
        loadingState.style.display = 'none';
        idleState.style.display = 'block';
        submitBtn.disabled = false;
        showAlert(
            'Connection Network Failure',
            `Could not communicate with verification service: ${err.message}`,
            'تعذر الاتصال بخدمة التحقق. يرجى التحقق من تشغيل الخادم والاتصال بالشبكة.'
        );
    }
}

// Render Results to UI
function renderResults(data) {
    const resultsContainer = document.getElementById('resultsContainer');
    document.getElementById('idleState').style.display = 'none';

    // Summary Card
    document.getElementById('resDigit').textContent = data.predicted_digit;
    document.getElementById('resConfidence').textContent = `${data.confidence.toFixed(2)}%`;
    document.getElementById('resLatency').textContent = `${data.inference_time_ms} ms`;
    document.getElementById('resPolarity').textContent = data.background_shade;
    document.getElementById('resSource').textContent = data.input_source;

    // Optical Matrix Preview
    document.getElementById('resPreviewImg').src = data.preview_image;
    document.getElementById('resBorderMean').textContent = `${data.perimeter_mean} / 255`;

    // Distribution Table
    const tbody = document.getElementById('distTableBody');
    tbody.innerHTML = '';

    data.distribution.forEach(item => {
        const tr = document.createElement('tr');
        if (item.digit === data.predicted_digit) {
            tr.className = 'winner-row';
        }

        const digitPrefix = item.digit === data.predicted_digit ? `[${item.digit}] *` : ` ${item.digit}  `;

        tr.innerHTML = `
            <td class="col-digit">${digitPrefix}</td>
            <td class="col-conf">${item.percentage.toFixed(2)}%</td>
            <td class="col-meter">
                <div class="meter-track">
                    <div class="meter-fill" style="width: ${item.percentage}%;"></div>
                </div>
            </td>
            <td class="col-raw">${item.probability.toFixed(6)}</td>
        `;
        tbody.appendChild(tr);
    });

    resultsContainer.style.display = 'flex';
}

// Alert Handlers
function showAlert(title, msgEn, msgAr) {
    const alertBox = document.getElementById('alertBox');
    document.getElementById('alertTitle').textContent = `[${title.toUpperCase()}]`;
    document.getElementById('alertMsgEn').textContent = msgEn;
    document.getElementById('alertMsgAr').textContent = msgAr;
    alertBox.style.display = 'block';
}

function hideAlert() {
    document.getElementById('alertBox').style.display = 'none';
}

// Reset Console
function resetAll() {
    clearSelectedFile();
    document.getElementById('urlInput').value = '';
    hideAlert();
    document.getElementById('loadingState').style.display = 'none';
    document.getElementById('resultsContainer').style.display = 'none';
    document.getElementById('idleState').style.display = 'block';
}
