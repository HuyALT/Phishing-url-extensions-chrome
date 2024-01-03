document.addEventListener('DOMContentLoaded', function () {
    const urlDisplay = document.getElementById('urlDisplay');
  
      chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
      const currentURL = tabs[0].url;
      urlDisplay.textContent = 'Current URL: ' + currentURL;
    });
  });