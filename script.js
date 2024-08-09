document.getElementById('years').addEventListener('input', function () {
    document.getElementById('yearsValue').textContent = this.value;
});

document.getElementById('prediction-form').addEventListener('submit', function (e) {
    e.preventDefault();
    const company = document.getElementById('company').value;
    const years = document.getElementById('years').value;

    // Fetch the company description (you'll need to implement this in your backend)
    fetch(`https://en.wikipedia.org/api/rest_v1/page/summary/${company}`)
        .then(response => response.json())
        .then(data => {
            document.getElementById('company-description').innerHTML = `<h2>${data.title}</h2><p>${data.extract}</p>`;
        });

    // Fetch and display prediction data (You'll need to implement the logic in the backend)
    // Display charts or other data in the #charts-container div
});
