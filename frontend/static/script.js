const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");
const predictBtn = document.getElementById("predictBtn");
const resultBox = document.getElementById("resultBox");

imageInput.addEventListener("change", function () {
    const file = imageInput.files[0];

    if (file) {
        previewImage.src = URL.createObjectURL(file);
        previewImage.style.display = "block";
    }
});

predictBtn.addEventListener("click", async function () {

    const file = imageInput.files[0];

    if (!file) {
        alert("Please upload an image.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {

        const response = await fetch("https://cnn-plant-disease-detection-system-7.onrender.com/", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        resultBox.style.display = "block";

        document.getElementById("disease").innerText = data.disease;
        document.getElementById("confidence").innerText = data.confidence + "%";
        document.getElementById("cause").innerText = data.cause;
        document.getElementById("treatment").innerText = data.treatment;
        document.getElementById("prevention").innerText = data.prevention;

    } catch (error) {
        console.error(error);
        alert("Error connecting to backend.");
    }
});
