// script for the index page
// preview functionality

function previewFile() {
    const previewContainer = document.getElementById('file-preview-container');
    const preview = document.getElementById('image-preview');
    const fileNameDisplay = document.getElementById('file-name-display');
    const fileInput = document.getElementById('image-upload');
    const file = fileInput.files[0];

    if (file) {
        const reader = new FileReader();

        // 1. Display filename
        fileNameDisplay.textContent = 'File: ' + file.name;
        
        // 2. Read file contents for preview
        reader.onloadend = function () {
            preview.src = reader.result;
            preview.style.display = 'block';
        }

        if (file.type.startsWith('image/')) {
            reader.readAsDataURL(file);
        } else {
            // Handle non-image files
            preview.src = '#';
            preview.style.display = 'none';
            fileNameDisplay.textContent += ' (Not an image file)';
        }

        // 3. Show the entire preview area
        previewContainer.style.display = 'block';

    } else {
        
        previewContainer.style.display = 'none';
        preview.src = '#';
        preview.style.display = 'none';
        fileNameDisplay.textContent = '';
    }
}