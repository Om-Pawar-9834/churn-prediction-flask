document.addEventListener("DOMContentLoaded", function () {

    const page = document.body.dataset.page;

    // =========================================================
    // PREDICTION PAGE
    // =========================================================

    if (page === "prediction") {
        setupPredictionPage();
    }

    // =========================================================
    // DASHBOARD PAGE
    // =========================================================

    if (page === "dashboard") {
        setupDashboard();
    }

});


// =============================================================
// PREDICTION PAGE
// =============================================================

function setupPredictionPage() {

    const form = document.getElementById("predictionForm");

    if (!form) {
        return;
    }

    const predictButton =
        document.getElementById("predictBtn");

    const predictText =
        document.getElementById("predictText");

    const spinner =
        document.getElementById("loadingSpinner");

    const exampleButton =
        document.getElementById("exampleBtn");


    // =========================================================
    // LOAD EXAMPLE
    // =========================================================

    if (exampleButton) {

        exampleButton.addEventListener("click", function () {

            const fields = {

                age: 32,

                gender: "Male",

                tenure: 12,

                usageFrequency: 18,

                supportCalls: 3,

                paymentDelay: 5,

                subscriptionType: "Standard",

                contractLength: "Monthly",

                totalSpend: 2500,

                lastInteraction: 7

            };


            setValue("age", fields.age);

            setValue("gender", fields.gender);

            setValue("tenure", fields.tenure);

            setValue(
                "usageFrequency",
                fields.usageFrequency
            );

            setValue(
                "supportCalls",
                fields.supportCalls
            );

            setValue(
                "paymentDelay",
                fields.paymentDelay
            );

            setValue(
                "subscriptionType",
                fields.subscriptionType
            );

            setValue(
                "contractLength",
                fields.contractLength
            );

            setValue(
                "totalSpend",
                fields.totalSpend
            );

            setValue(
                "lastInteraction",
                fields.lastInteraction
            );

        });

    }


    // =========================================================
    // FORM SUBMIT
    // =========================================================

    form.addEventListener("submit", async function (event) {

        event.preventDefault();


        // -----------------------------------------------------
        // BUTTON LOADING STATE
        // -----------------------------------------------------

        if (predictButton) {
            predictButton.disabled = true;
        }

        if (predictText) {
            predictText.textContent = "Analyzing...";
        }

        if (spinner) {
            spinner.classList.remove("hidden");
        }


        // -----------------------------------------------------
        // COLLECT FORM DATA
        // -----------------------------------------------------

        const formData = new FormData(form);


        const data = {

            "Age":
                formData.get("Age"),

            "Gender":
                formData.get("Gender"),

            "Tenure":
                formData.get("Tenure"),

            "Usage Frequency":
                formData.get("Usage Frequency"),

            "Support Calls":
                formData.get("Support Calls"),

            "Payment Delay":
                formData.get("Payment Delay"),

            "Subscription Type":
                formData.get("Subscription Type"),

            "Contract Length":
                formData.get("Contract Length"),

            "Total Spend":
                formData.get("Total Spend"),

            "Last Interaction":
                formData.get("Last Interaction")

        };


        try {

            // -------------------------------------------------
            // SEND DATA TO FLASK
            // -------------------------------------------------

            const response = await fetch("/predict", {

                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify(data)

            });


            const result =
                await response.json();


            // -------------------------------------------------
            // ERROR
            // -------------------------------------------------

            if (!response.ok || !result.success) {

                throw new Error(
                    result.error ||
                    "Prediction failed."
                );

            }


            // -------------------------------------------------
            // DISPLAY RESULT
            // -------------------------------------------------

            showPredictionResult(result);


        } catch (error) {

            console.error(
                "Prediction error:",
                error
            );

            alert(
                "Prediction Error: " +
                error.message
            );


        } finally {

            if (predictButton) {
                predictButton.disabled = false;
            }

            if (predictText) {
                predictText.textContent =
                    "Predict Churn";
            }

            if (spinner) {
                spinner.classList.add("hidden");
            }

        }

    });

}


// =============================================================
// SET FORM VALUE
// =============================================================

function setValue(id, value) {

    const element =
        document.getElementById(id);

    if (element) {
        element.value = value;
    }

}


// =============================================================
// SHOW PREDICTION RESULT
// =============================================================

function showPredictionResult(result) {

    const placeholder =
        document.getElementById(
            "resultPlaceholder"
        );

    const content =
        document.getElementById(
            "resultContent"
        );


    if (placeholder) {
        placeholder.classList.add("hidden");
    }

    if (content) {
        content.classList.remove("hidden");
    }


    // ---------------------------------------------------------
    // PREDICTION
    // ---------------------------------------------------------

    const predictionText =
        document.getElementById(
            "predictionText"
        );

    if (predictionText) {

        predictionText.textContent =
            result.prediction;

    }


    // ---------------------------------------------------------
    // PROBABILITY
    // ---------------------------------------------------------

    const probability =
        Number(result.probability) || 0;


    const probabilityText =
        document.getElementById(
            "probabilityText"
        );

    if (probabilityText) {

        probabilityText.textContent =
            probability.toFixed(2) + "%";

    }


    const probabilityBar =
        document.getElementById(
            "probabilityBar"
        );

    if (probabilityBar) {

        probabilityBar.style.width =
            Math.min(
                Math.max(probability, 0),
                100
            ) + "%";

    }


    // ---------------------------------------------------------
    // RISK
    // ---------------------------------------------------------

    const riskText =
        document.getElementById(
            "riskText"
        );

    if (riskText) {

        riskText.textContent =
            result.risk;

    }


    const riskBadge =
        document.getElementById(
            "riskBadge"
        );

    if (riskBadge) {

        riskBadge.textContent =
            result.risk;

        riskBadge.className =
            "risk-badge " +
            String(result.risk)
                .toLowerCase();

    }


    // ---------------------------------------------------------
    // TIME
    // ---------------------------------------------------------

    const timeText =
        document.getElementById(
            "timeText"
        );

    if (timeText) {

        timeText.textContent =
            result.timestamp || "Just now";

    }


    // ---------------------------------------------------------
    // MESSAGE
    // ---------------------------------------------------------

    let message =
        "Customer has a relatively low churn risk.";


    if (result.risk === "Medium") {

        message =
            "Customer shows moderate churn risk. Consider proactive engagement.";

    }


    if (result.risk === "High") {

        message =
            "Customer has a high churn probability. Immediate retention action may be useful.";

    }


    const predictionMessage =
        document.getElementById(
            "predictionMessage"
        );


    if (predictionMessage) {

        predictionMessage.textContent =
            message;

    }

}


// =============================================================
// DASHBOARD
// =============================================================

function setupDashboard() {

    loadDashboard();


    // ---------------------------------------------------------
    // REFRESH BUTTON
    // ---------------------------------------------------------

    const refreshButton =
        document.getElementById(
            "refreshDashboard"
        );


    if (refreshButton) {

        refreshButton.addEventListener(
            "click",
            async function () {

                refreshButton.disabled = true;

                const oldText =
                    refreshButton.textContent;

                refreshButton.textContent =
                    "↻ Refreshing...";


                await loadDashboard();


                refreshButton.disabled = false;

                refreshButton.textContent =
                    oldText;

            }
        );

    }


    // ---------------------------------------------------------
    // CLEAR HISTORY
    // ---------------------------------------------------------

    const clearButton =
        document.getElementById(
            "clearHistory"
        );


    if (clearButton) {

        clearButton.addEventListener(
            "click",
            clearHistory
        );

    }

}


// =============================================================
// LOAD DASHBOARD DATA
// =============================================================

async function loadDashboard() {

    try {

        const response =
            await fetch(
                "/api/dashboard",
                {
                    method: "GET",
                    cache: "no-store"
                }
            );


        const data =
            await response.json();


        if (!response.ok || !data.success) {

            throw new Error(
                data.error ||
                "Dashboard data could not be loaded."
            );

        }


        updateDashboard(data);


    } catch (error) {

        console.error(
            "Dashboard error:",
            error
        );


        showDashboardError(
            error.message
        );

    }

}


// =============================================================
// UPDATE DASHBOARD
// =============================================================

function updateDashboard(data) {

    const stats =
        data.stats || {};


    // ---------------------------------------------------------
    // STAT CARDS
    // ---------------------------------------------------------

    setText(
        "totalPredictions",
        stats.total || 0
    );


    setText(
        "highRisk",
        stats.high || 0
    );


    setText(
        "mediumRisk",
        stats.medium || 0
    );


    setText(
        "lowRisk",
        stats.low || 0
    );


    const averageProbability =
        Number(
            stats.average_probability
        ) || 0;


    setText(
        "averageProbability",
        averageProbability.toFixed(2) + "%"
    );


    // ---------------------------------------------------------
    // RISK PERCENTAGES
    // ---------------------------------------------------------

    const total =
        Number(stats.total) || 0;


    let highPercentage = 0;

    let mediumPercentage = 0;

    let lowPercentage = 0;


    if (total > 0) {

        highPercentage =
            (Number(stats.high) / total) * 100;

        mediumPercentage =
            (Number(stats.medium) / total) * 100;

        lowPercentage =
            (Number(stats.low) / total) * 100;

    }


    // ---------------------------------------------------------
    // PERCENTAGE TEXT
    // ---------------------------------------------------------

    setText(
        "highPercentage",
        highPercentage.toFixed(1) + "%"
    );


    setText(
        "mediumPercentage",
        mediumPercentage.toFixed(1) + "%"
    );


    setText(
        "lowPercentage",
        lowPercentage.toFixed(1) + "%"
    );


    // ---------------------------------------------------------
    // PROGRESS BARS
    // ---------------------------------------------------------

    setWidth(
        "highBar",
        highPercentage
    );


    setWidth(
        "mediumBar",
        mediumPercentage
    );


    setWidth(
        "lowBar",
        lowPercentage
    );


    // ---------------------------------------------------------
    // LATEST PREDICTION
    // ---------------------------------------------------------

    updateLatestPrediction(
        data.latest || null
    );


    // ---------------------------------------------------------
    // HISTORY
    // ---------------------------------------------------------

    updateHistory(
        data.history || []
    );

}


// =============================================================
// LATEST PREDICTION
// =============================================================

function updateLatestPrediction(latest) {

    const container =
        document.getElementById(
            "latestPrediction"
        );


    if (!container) {
        return;
    }


    // ---------------------------------------------------------
    // NO DATA
    // ---------------------------------------------------------

    if (!latest) {

        container.innerHTML = `

            <div class="empty-state">

                <div class="empty-icon">
                    ML
                </div>

                <p>
                    No predictions yet.
                </p>

                <a href="/">
                    Make your first prediction →
                </a>

            </div>

        `;

        return;

    }


    const probability =
        Number(
            latest.probability
        ) || 0;


    const risk =
        latest.risk || "Low";


    container.innerHTML = `

        <div class="latest-result">

            <div class="latest-result-top">

                <div>

                    <span>
                        Churn Prediction
                    </span>

                    <h3>
                        ${escapeHtml(
                            latest.prediction
                        )}
                    </h3>

                </div>

                <span
                    class="risk-badge ${String(
                        risk
                    ).toLowerCase()}"
                >
                    ${escapeHtml(risk)}
                </span>

            </div>


            <div class="latest-probability">

                <span>
                    Churn Probability
                </span>

                <strong>
                    ${probability.toFixed(2)}%
                </strong>

            </div>


            <div class="mini-progress">

                <div
                    style="width: ${
                        Math.min(
                            Math.max(
                                probability,
                                0
                            ),
                            100
                        )
                    }%"
                ></div>

            </div>


            <p class="latest-time">

                ${escapeHtml(
                    latest.timestamp || ""
                )}

            </p>

        </div>

    `;

}


// =============================================================
// HISTORY TABLE
// =============================================================

function updateHistory(history) {

    const table =
        document.getElementById(
            "historyTable"
        );


    const empty =
        document.getElementById(
            "emptyHistory"
        );


    const count =
        document.getElementById(
            "historyCount"
        );


    if (!table) {
        return;
    }


    history =
        Array.isArray(history)
            ? history
            : [];


    // ---------------------------------------------------------
    // HISTORY COUNT
    // ---------------------------------------------------------

    if (count) {

        count.textContent =
            history.length +
            (
                history.length === 1
                    ? " record"
                    : " records"
            );

    }


    // ---------------------------------------------------------
    // EMPTY HISTORY
    // ---------------------------------------------------------

    if (!history.length) {

        table.innerHTML = "";


        if (empty) {
            empty.style.display = "block";
        }


        return;

    }


    if (empty) {
        empty.style.display = "none";
    }


    // ---------------------------------------------------------
    // CREATE TABLE ROWS
    // ---------------------------------------------------------

    table.innerHTML =
        history
            .map(function (item) {

                const customer =
                    item.customer || {};


                const probability =
                    Number(
                        item.probability
                    ) || 0;


                const risk =
                    item.risk || "Low";


                /*
                 * IMPORTANT:
                 * The Flask backend should provide
                 * a unique "id" for every prediction.
                 */

                const predictionId =
                    item.id ||
                    item.timestamp ||
                    "";


                return `

                    <tr>

                        <!-- TIME -->
                        <td>

                            ${escapeHtml(
                                item.timestamp || "-"
                            )}

                        </td>


                        <!-- PREDICTION -->
                        <td>

                            <strong>

                                ${escapeHtml(
                                    item.prediction
                                )}

                            </strong>

                        </td>


                        <!-- PROBABILITY -->
                        <td>

                            <div
                                class="table-probability"
                            >

                                <span>

                                    ${probability.toFixed(2)}%

                                </span>


                                <div
                                    class="mini-progress"
                                >

                                    <div
                                        style="width: ${
                                            Math.min(
                                                Math.max(
                                                    probability,
                                                    0
                                                ),
                                                100
                                            )
                                        }%"
                                    ></div>

                                </div>

                            </div>

                        </td>


                        <!-- RISK -->
                        <td>

                            <span
                                class="risk-badge ${String(
                                    risk
                                ).toLowerCase()}"
                            >

                                ${escapeHtml(risk)}

                            </span>

                        </td>


                        <!-- CUSTOMER -->
                        <td>

                            <div
                                class="customer-summary"
                            >

                                <strong>

                                    ${escapeHtml(
                                        customer.Gender ||
                                        "Customer"
                                    )}

                                </strong>


                                <small>

                                    Age:
                                    ${escapeHtml(
                                        customer.Age ??
                                        "-"
                                    )}

                                    |

                                    Tenure:
                                    ${escapeHtml(
                                        customer.Tenure ??
                                        "-"
                                    )}

                                </small>

                            </div>

                        </td>


                        <!-- DELETE -->
                        <td>

                            <button
                                type="button"
                                class="delete-prediction-btn"
                                data-id="${escapeHtml(
                                    predictionId
                                )}"
                                title="Delete this prediction"
                            >
                                Delete
                            </button>

                        </td>

                    </tr>

                `;

            })
            .join("");


    // ---------------------------------------------------------
    // DELETE BUTTON EVENTS
    // ---------------------------------------------------------

    const deleteButtons =
        document.querySelectorAll(
            ".delete-prediction-btn"
        );


    deleteButtons.forEach(function (button) {

        button.addEventListener(
            "click",
            function () {

                const id =
                    button.dataset.id;


                deletePrediction(
                    id,
                    button
                );

            }
        );

    });

}


// =============================================================
// DELETE ONE PREDICTION
// =============================================================

async function deletePrediction(
    predictionId,
    button
) {

    if (!predictionId) {

        alert(
            "Unable to identify this prediction."
        );

        return;

    }


    // ---------------------------------------------------------
    // CONFIRMATION
    // ---------------------------------------------------------

    const confirmed =
        confirm(
            "Are you sure you want to delete this prediction?"
        );


    if (!confirmed) {
        return;
    }


    try {

        // -----------------------------------------------------
        // BUTTON LOADING
        // -----------------------------------------------------

        if (button) {

            button.disabled = true;

            button.textContent =
                "Deleting...";

        }


        // -----------------------------------------------------
        // DELETE REQUEST
        // -----------------------------------------------------

        const response =
            await fetch(
                "/delete-prediction/" +
                encodeURIComponent(
                    predictionId
                ),
                {
                    method: "DELETE"
                }
            );


        const result =
            await response.json();


        if (!response.ok || !result.success) {

            throw new Error(
                result.error ||
                "Unable to delete prediction."
            );

        }


        // -----------------------------------------------------
        // REFRESH DASHBOARD
        // -----------------------------------------------------

        await loadDashboard();


    } catch (error) {

        console.error(
            "Delete prediction error:",
            error
        );


        alert(
            "Delete Error: " +
            error.message
        );


        if (button) {

            button.disabled = false;

            button.textContent =
                "Delete";

        }

    }

}


// =============================================================
// CLEAR ALL HISTORY
// =============================================================

async function clearHistory() {

    const confirmed =
        confirm(
            "Are you sure you want to clear ALL prediction history?\n\nThis action cannot be undone."
        );


    if (!confirmed) {
        return;
    }


    try {

        const response =
            await fetch(
                "/clear-history",
                {
                    method: "POST"
                }
            );


        const result =
            await response.json();


        if (!response.ok || !result.success) {

            throw new Error(
                result.error ||
                "Unable to clear prediction history."
            );

        }


        // Refresh dashboard

        await loadDashboard();


    } catch (error) {

        console.error(
            "Clear history error:",
            error
        );


        alert(
            "Unable to clear prediction history.\n\n" +
            error.message
        );

    }

}


// =============================================================
// SET TEXT
// =============================================================

function setText(
    elementId,
    value
) {

    const element =
        document.getElementById(
            elementId
        );


    if (element) {

        element.textContent =
            value;

    }

}


// =============================================================
// SET WIDTH
// =============================================================

function setWidth(
    elementId,
    value
) {

    const element =
        document.getElementById(
            elementId
        );


    if (element) {

        element.style.width =
            Math.min(
                Math.max(
                    Number(value) || 0,
                    0
                ),
                100
            ) + "%";

    }

}


// =============================================================
// DASHBOARD ERROR
// =============================================================

function showDashboardError(message) {

    console.error(
        "Dashboard:",
        message
    );

}


// =============================================================
// HTML ESCAPE
// =============================================================

function escapeHtml(value) {

    return String(
        value ?? ""
    )

        .replaceAll(
            "&",
            "&amp;"
        )

        .replaceAll(
            "<",
            "&lt;"
        )

        .replaceAll(
            ">",
            "&gt;"
        )

        .replaceAll(
            '"',
            "&quot;"
        )

        .replaceAll(
            "'",
            "&#039;"
        );

}