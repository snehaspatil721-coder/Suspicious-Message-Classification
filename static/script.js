
const form = document.getElementById("messageForm");
const messageInput = document.getElementById("message");
const classifyButton = document.getElementById("classifyButton");
const loading = document.getElementById("loading");
const result = document.getElementById("result");
const resultTitle = document.getElementById("resultTitle");
const resultText = document.getElementById("resultText");

form.addEventListener("submit", async function (event) {
    event.preventDefault();

    const message = messageInput.value.trim();

    if (!message) {
        showResult("error", "Message required", "Please enter a message.");
        return;
    }

    classifyButton.disabled = true;
    loading.classList.remove("hidden");
    result.classList.add("hidden");

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message: message })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Unable to classify this message.");
        }

        if (data.prediction === "spam") {
            showResult(
                "spam",
                "⚠️ SPAM / SUSPICIOUS MESSAGE",
                "The machine learning model classified this message as spam."
            );
        } else {
            showResult(
                "ham",
                "✅ HAM / LIKELY LEGITIMATE",
                "The machine learning model classified this message as ham."
            );
        }
    } catch (error) {
        showResult("error", "Classification failed", error.message);
    } finally {
        classifyButton.disabled = false;
        loading.classList.add("hidden");
    }
});

function showResult(type, title, text) {
    result.className = "result " + type;
    resultTitle.textContent = title;
    resultText.textContent = text;
    result.classList.remove("hidden");
}


