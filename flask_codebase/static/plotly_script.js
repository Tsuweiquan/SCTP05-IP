// plotly_script.js

// Function to render the Plotly figure
function renderPlot() {
    const figureData = JSON.parse(document.getElementById('plotly-data').textContent);
    const layout = { title: 'Your Chart Title' }; // Customize layout as needed
    Plotly.newPlot('plotly-figure', figureData.data, layout);
}

// Call renderPlot when the window loads
window.onload = renderPlot;