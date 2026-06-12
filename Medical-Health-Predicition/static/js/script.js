// Medical Health Prediction System - JavaScript

document.addEventListener("DOMContentLoaded", function () {
  // Initialize symptom search
  initSymptomSearch();

  // Initialize symptom counter
  initSymptomCounter();

  // Initialize form validation
  initFormValidation();
});

// Symptom Search Functionality
function initSymptomSearch() {
  const searchInput = document.getElementById("symptom-search");
  if (!searchInput) return;

  searchInput.addEventListener("input", function (e) {
    const searchTerm = e.target.value.toLowerCase();
    const symptoms = document.querySelectorAll(".symptom-checkbox");

    symptoms.forEach(function (symptom) {
      const label = symptom
        .querySelector(".symptom-label")
        .textContent.toLowerCase();
      if (label.includes(searchTerm)) {
        symptom.style.display = "flex";
      } else {
        symptom.style.display = "none";
      }
    });
  });
}

// Symptom Counter
function initSymptomCounter() {
  const checkboxes = document.querySelectorAll(
    '.symptom-checkbox input[type="checkbox"]'
  );
  const countDisplay = document.getElementById("selected-count");

  if (!countDisplay || checkboxes.length === 0) return;

  function updateCount() {
    const checked = document.querySelectorAll(
      '.symptom-checkbox input[type="checkbox"]:checked'
    );
    countDisplay.textContent = checked.length;
  }

  checkboxes.forEach(function (checkbox) {
    checkbox.addEventListener("change", updateCount);
  });
}

// Form Validation
function initFormValidation() {
  const form = document.getElementById("prediction-form");
  if (!form) return;

  form.addEventListener("submit", function (e) {
    const checked = document.querySelectorAll(
      '.symptom-checkbox input[type="checkbox"]:checked'
    );

    if (checked.length === 0) {
      e.preventDefault();
      alert("Please select at least one symptom before predicting.");
      return false;
    }

    // Show loading state
    const submitBtn = form.querySelector('button[type="submit"]');
    if (submitBtn) {
      submitBtn.innerHTML =
        '<i class="fas fa-spinner fa-spin"></i> Predicting...';
      submitBtn.disabled = true;
    }
  });
}

// Utility function to format symptom names
function formatSymptomName(symptom) {
  return symptom.replace(/_/g, " ").replace(/\b\w/g, function (l) {
    return l.toUpperCase();
  });
}

// Clear all selected symptoms
function clearSymptoms() {
  const checkboxes = document.querySelectorAll(
    '.symptom-checkbox input[type="checkbox"]'
  );
  checkboxes.forEach(function (checkbox) {
    checkbox.checked = false;
  });

  const countDisplay = document.getElementById("selected-count");
  if (countDisplay) {
    countDisplay.textContent = "0";
  }
}

// Select random symptoms for testing
function selectRandomSymptoms(count) {
  clearSymptoms();

  const checkboxes = Array.from(
    document.querySelectorAll('.symptom-checkbox input[type="checkbox"]')
  );
  const shuffled = checkboxes.sort(function () {
    return 0.5 - Math.random();
  });
  const selected = shuffled.slice(0, count);

  selected.forEach(function (checkbox) {
    checkbox.checked = true;
  });

  const countDisplay = document.getElementById("selected-count");
  if (countDisplay) {
    countDisplay.textContent = count;
  }
}
