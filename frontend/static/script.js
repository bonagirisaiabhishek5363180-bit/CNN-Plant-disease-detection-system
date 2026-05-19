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

    const backendOrigin = window.location.origin === "null" ? "https://cnn-plant-disease-detection-system-8.onrender.com/" : window.location.origin;
    const endpoint = `${backendOrigin}/`;
    console.log("Sending request to backend:", endpoint);

    try {
        const response = await fetch(endpoint, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => null);
            const message = errorData?.error || `Server error ${response.status}`;
            throw new Error(message);
        }

        const data = await response.json();

        resultBox.style.display = "block";

        document.getElementById("disease").innerText = data.disease;
        document.getElementById("confidence").innerText = data.confidence + "%";
        document.getElementById("cause").innerText = data.cause || "N/A";
        document.getElementById("treatment").innerText = data.treatment || "N/A";
        document.getElementById("prevention").innerText = data.prevention || "N/A";
    } catch (error) {
        console.error(error);
        alert(`Error connecting to backend: ${error.message}`);
    }
});
