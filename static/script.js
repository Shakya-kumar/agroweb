document.addEventListener('DOMContentLoaded', () => {
    const recommendBtn = document.getElementById('recommendBtn');
    const requirementsBtn = document.getElementById('requirementsBtn');
    const resultSection = document.getElementById('resultSection');
    const recommendationResult = document.getElementById('recommendationResult');
    const requirementsResult = document.getElementById('requirementsResult');
    const requirementsList = document.getElementById('requirementsList');

    // Function to get form data
    const getFormData = () => {
        const data = {
            N: document.getElementById('N').value,
            P: document.getElementById('P').value,
            K: document.getElementById('K').value,
            temperature: document.getElementById('temperature').value,
            humidity: document.getElementById('humidity').value,
            ph: document.getElementById('ph').value,
            rainfall: document.getElementById('rainfall').value
        };

        // Convert all values to numbers and validate
        for (const key in data) {
            const value = parseFloat(data[key]);
            if (isNaN(value)) {
                alert(`Please enter a valid number for ${key}`);
                return null;
            }
            data[key] = value;
        }

        return data;
    };

    // Function to validate form data
    const validateFormData = (data) => {
        if (!data) return false;

        // Check if any value is missing
        for (const key in data) {
            if (data[key] === undefined || data[key] === null) {
                alert(`Please enter a value for ${key}`);
                return false;
            }
        }

        // Validate pH value
        if (data.ph < 1 || data.ph > 14) {
            alert('pH value must be between 1 and 14');
            return false;
        }

        return true;
    };

    // Function to handle API errors
    const handleError = (error) => {
        console.error('Error:', error);
        alert('An error occurred. Please try again.');
    };

    // Function to display recommendation results
    const displayRecommendation = (data) => {
        document.getElementById('recommendedCrop').textContent = data.recommended_crop;
        document.getElementById('hindiName').textContent = data.hindi_name;
        recommendationResult.style.display = 'block';
        requirementsResult.style.display = 'none';
        resultSection.style.display = 'block';
    };

    // Function to display requirements results
    const displayRequirements = (data) => {
        requirementsList.innerHTML = '';
        data.requirements.forEach(req => {
            const item = document.createElement('div');
            item.className = 'requirement-item';
            
            const adjustment = req.adjustment;
            const adjustmentText = adjustment > 0 
                ? `Need to add ${Math.abs(adjustment).toFixed(2)} ${req.unit}`
                : adjustment < 0 
                    ? `Excess of ${Math.abs(adjustment).toFixed(2)} ${req.unit}`
                    : 'Optimal';

            item.innerHTML = `
                <span class="requirement-parameter">${req.parameter}</span>
                <span class="requirement-value">
                    Current: ${req.current_value.toFixed(2)} ${req.unit}<br>
                    Required: ${req.required_value.toFixed(2)} ${req.unit}<br>
                    ${adjustmentText}
                </span>
            `;
            requirementsList.appendChild(item);
        });

        document.getElementById('recommendedCrop').textContent = data.crop;
        document.getElementById('hindiName').textContent = data.hindi_name;
        recommendationResult.style.display = 'block';
        requirementsResult.style.display = 'block';
        resultSection.style.display = 'block';
    };

    // Event listener for recommendation button
    recommendBtn.addEventListener('click', async () => {
        const formData = getFormData();
        if (!validateFormData(formData)) return;

        try {
            const response = await fetch('/api/recommend', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to get recommendation');
            }

            const data = await response.json();
            displayRecommendation(data);
        } catch (error) {
            handleError(error);
        }
    });

    // Event listener for requirements button
    requirementsBtn.addEventListener('click', async () => {
        const formData = getFormData();
        if (!validateFormData(formData)) return;

        const cropName = document.getElementById('cropName').value.trim();
        if (!cropName) {
            alert('Please enter a crop name');
            return;
        }

        try {
            const response = await fetch('/api/crop-requirements', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    ...formData,
                    crop_name: cropName
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to get crop requirements');
            }

            const data = await response.json();
            if (data.error) {
                alert(data.error);
                return;
            }
            displayRequirements(data);
        } catch (error) {
            handleError(error);
        }
    });
}); 