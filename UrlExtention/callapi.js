const checkbutton = document.getElementById('CheckButton')
const legitimate = document.getElementById('Legitimate')
const phishing = document.getElementById('Phishing')
const buttonCheckURL = document.getElementById('button-check-url-submit')
const resultbg = document.getElementById('result')

checkbutton.addEventListener('click', function () {
  chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
    const currentURL = tabs[0].url;
    const urlapi = 'http://127.0.0.1:5000/api/urlcheck?url='+currentURL

    const headers = new Headers();
    headers.append('Content-Type', 'application/json');


    // Now you can use 'requestOptions' to make your POST request
    fetch(urlapi,{ method: 'GET', headers: headers})
    .then(response => {
      console.log(response.clone().json())
      return response.clone().json(); // Parse the response as JSON
    })
    .then(responseData => {
      // Handle the response data
      console.log(responseData.data);
      if (responseData.data=='legitimate') {
        legitimate.style.display = 'block'
        resultbg.style.backgroundColor = 'green'
      }
      else{
        phishing.style.display = 'block'
        resultbg.style.backgroundColor = 'red'
      }
    });
  });
});

buttonCheckURL.addEventListener('click', function() {
  const inputURL = document.getElementById('InputURL').value.toString()
  const urlapi = 'http://127.0.0.1:5000/api/urlcheck?url='+inputURL

  const headers = new Headers();
  headers.append('Content-Type', 'application/json');

  // Now you can use 'requestOptions' to make your POST request
  fetch(urlapi,{ method: 'GET', headers: headers})
  .then(response => {
    console.log(response.clone().json())
    return response.clone().json(); // Parse the response as JSON
  })
  .then(responseData => {
    // Handle the response data
    console.log(responseData.data);
    if (responseData.data=='legitimate') {
      legitimate.style.display = 'block'
    }
    else{
      phishing.style.display = 'block'
    }
  });
});

document.addEventListener('DOMContentLoaded', function(){
  legitimate.style.display = 'none'
  phishing.style.display = 'none'
  resultbg.style.backgroundColor = 'none'
});
