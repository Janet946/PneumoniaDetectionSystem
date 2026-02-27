const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const previewImg = document.getElementById("previewImg");
const previewPlaceholder = document.getElementById("previewPlaceholder");
const clearBtn = document.getElementById("clearBtn");

if (dropzone && fileInput) {
  function showPreview(file) {
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    previewImg.style.display = "block";
    previewPlaceholder.style.display = "none";
  }

  function clearPreview() {
    fileInput.value = "";
    previewImg.src = "";
    previewImg.style.display = "none";
    previewPlaceholder.style.display = "block";
  }

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      fileInput.files = e.dataTransfer.files;
      showPreview(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) showPreview(fileInput.files[0]);
  });

  if (clearBtn) clearBtn.addEventListener("click", clearPreview);
}
