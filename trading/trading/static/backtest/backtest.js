document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("backtest-form");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const ticker = document.getElementById("ticker").value;
    const start_date = document.getElementById("start-date").value;
    const end_date = document.getElementById("end-date").value;

    try {
      const response = await fetch("/run-backtest/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-CSRFToken": getCookie("csrftoken"),
        },
        body: JSON.stringify({ ticker, start_date, end_date }),
      });

      const data = await response.json();
      document.getElementById("result").textContent = JSON.stringify(data, null, 2);
    } catch (err) {
      console.error("백테스트 오류:", err);
    }
  });

  // CSRF 토큰 가져오기
  function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== "") {
      const cookies = document.cookie.split(";");
      for (const cookie of cookies) {
        const trimmed = cookie.trim();
        if (trimmed.startsWith(name + "=")) {
          cookieValue = decodeURIComponent(trimmed.substring(name.length + 1));
          break;
        }
      }
    }
    return cookieValue;
  }
});
