/*  ==========================================
    SHOW UPLOADED IMAGE
* ========================================== */
function readURL(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function (e) {
            $('#imageResult')
                .attr('src', e.target.result);
        };
        reader.readAsDataURL(input.files[0]);
    }
}

$(function () {
    $('#upload').on('change', function () {
        readURL(input);
    });
});

/*  ==========================================
    SHOW UPLOADED IMAGE NAME
* ========================================== */
var input = document.getElementById( 'upload' );
var infoArea = document.getElementById( 'upload-label' );

input.addEventListener( 'change', showFileName );
function showFileName( event ) {
  var input = event.srcElement;
  var fileName = input.files[0].name;
  infoArea.textContent = 'File name: ' + fileName;
}

/*  ==========================================
    CALL REST API AND RENDER RESULT
* ========================================== */
var predictForm = document.getElementById('predict-form');
var predictionResult = document.getElementById('prediction-result');

if (predictForm) {
    predictForm.addEventListener('submit', async function (event) {
        event.preventDefault();

        if (!input.files || !input.files[0]) {
            predictionResult.textContent = 'Please select an image first.';
            return;
        }

        var formData = new FormData();
        formData.append('file', input.files[0]);

        predictionResult.textContent = 'Predicting...';

        try {
            var response = await fetch('/api/v1/predict', {
                method: 'POST',
                body: formData
            });

            var data = await response.json();

            if (!response.ok) {
                predictionResult.textContent = 'Error: ' + (data.error || 'prediction failed');
                return;
            }

            predictionResult.textContent = 'Breed: ' + data.breed + ' (confidence: ' + data.confidence + ')';
        } catch (error) {
            predictionResult.textContent = 'Request failed. Please try again.';
        }
    });
}