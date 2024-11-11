// Existing script
let savedpasttext = []; // Variable to store the message
let savedpastresponse = []; // Variable to store the message

// Section: get the Id of the talking container
const messagesContainer = document.getElementById('messages-container');
const messageForm = document.getElementById('message-form');
const messageInput = document.getElementById('message-input');

// Section: function to create the dialogue window
const addMessage = (message, role, imgSrc) => {
  // create elements in the dialogue window
  const messageElement = document.createElement('div');
  const textElement = document.createElement('p');
  messageElement.className = `message ${role}`;
  const imgElement = document.createElement('img');
  imgElement.src = imgSrc;
  // append the image and message to the message element
  messageElement.appendChild(imgElement);
  textElement.innerText = message;
  messageElement.appendChild(textElement);
  messagesContainer.appendChild(messageElement);
  // create the ending of the message
  const clearDiv = document.createElement("div");
  clearDiv.style.clear = "both";
  messagesContainer.appendChild(clearDiv);
};

// Section: Calling the model
const sendMessage = async (message) => {
  addMessage(message, 'user', '/static/user.png');
  // Loading animation
  const loadingElement = document.createElement('div');
  const loadingtextElement = document.createElement('p');
  loadingElement.className = `loading-animation`;
  loadingtextElement.className = `loading-text`;
  loadingtextElement.innerText = 'Loading....Please wait';
  messagesContainer.appendChild(loadingElement);
  messagesContainer.appendChild(loadingtextElement);

  async function makePostRequest(msg) {
    const url = 'http://127.0.0.1:5000/chatbot';  // Corrected URL
    const requestBody = {
      prompt: msg
    };
  
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
      });
  
      const data = await response.text();
      // Handle the response data here
      console.log(data);
      return data;
    } catch (error) {
      // Handle any errors that occurred during the request
      console.error('Error:', error);
      return error;
    }
  }
  
  const res = await makePostRequest(message);
  
  const data = {"response": res};
  
  // Deleting the loading animation
  const loadanimation = document.querySelector('.loading-animation');
  const loadtxt = document.querySelector('.loading-text');
  loadanimation.remove();
  loadtxt.remove();

  if (data.error) {
    // Handle the error here
    const errorMessage = JSON.stringify(data);
    addMessage(errorMessage, 'error', '/static/Error.png');
  } else {
    // Process the normal response here
    const responseMessage = data['response'];
    addMessage(responseMessage, 'aibot', '/static/Bot_logo.png');
  }
};

// Section: Button to submit to the model and get the response
messageForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (message !== '') {
    messageInput.value = '';
    await sendMessage(message);
  }
});