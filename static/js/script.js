

function previewFile() {
    const previewContainer = document.getElementById('file-preview-container');
    const fileInput = document.getElementById('image-upload');
    const imagePreview = document.getElementById('image-preview');
    const fileNameDisplay = document.getElementById('file-name-display'); // For the actual file name
    const fileNameLabel = document.getElementById('file-name-display-label'); // For "No file selected"

    if (fileInput.files && fileInput.files[0]) {
        const reader = new FileReader();
        reader.onload = function (e) {
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
        };
        reader.readAsDataURL(fileInput.files[0]);

        fileNameDisplay.textContent = fileInput.files[0].name; // Update hidden name
        fileNameLabel.textContent = fileInput.files[0].name; // Update visible name label
        previewContainer.style.display = 'block';
    } else {
        imagePreview.src = '#';
        imagePreview.style.display = 'none';
        fileNameDisplay.textContent = '';
        fileNameLabel.textContent = 'No file selected.';
        previewContainer.style.display = 'none';
    }
}
