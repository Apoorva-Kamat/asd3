// Access the camera
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');
const captureButton = document.getElementById('capture');
const resultText = document.getElementById('result');
const uploadForm = document.getElementById('uploadForm');
const fileInput = document.getElementById('fileInput');

// Function to display predictions
function displayPredictions(predictions) {
    document.getElementById('nnPrediction').textContent = `Neural Network: ${predictions['Neural Network'].result}`;
    document.getElementById('dtPrediction').textContent = `Decision Tree: ${predictions['Decision Tree'].result}`;
    document.getElementById('lrPrediction').textContent = `Logistic Regression: ${predictions['Logistic Regression'].result}`;
}

// Function to display accuracy graphs
function displayAccuracyGraphs(nnAccuracy, dtAccuracy, lrAccuracy) {
    const data = [
        {
            x: ['Neural Network', 'Decision Tree', 'Logistic Regression'],
            y: [nnAccuracy, dtAccuracy, lrAccuracy],
            type: 'bar'
        }
    ];
    
    Plotly.newPlot('accuracyGraphs', data);
}

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    });

captureButton.addEventListener('click', () => {
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataURL = canvas.toDataURL('image/jpeg');
    const base64Data = dataURL.split(',')[1];

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: base64Data }),
    })
    .then(response => response.json())
    .then(data => {
        displayPredictions(data.result);
        displayAccuracyGraphs(
            data.result['Neural Network'].accuracy,
            data.result['Decision Tree'].accuracy,
            data.result['Logistic Regression'].accuracy
        );
    })
    .catch(error => console.error('Error:', error));
});

uploadForm.addEventListener('submit', (event) => {
    event.preventDefault();
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    fetch('/upload', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        displayPredictions(data.result);
        displayAccuracyGraphs(
            data.result['Neural Network'].accuracy,
            data.result['Decision Tree'].accuracy,
            data.result['Logistic Regression'].accuracy
        );
    })
    .catch(error => console.error('Error:', error));
});
