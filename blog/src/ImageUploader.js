// src/ImageUploader.js

import React, { useState } from 'react';

const ImageUploader = () => {
  const [serverUrl, setServerUrl] = useState(null);
  const [enteredKey, setEnteredKey] = useState('');

  const dragEnter = (ev) => {
    ev.preventDefault();
  };

  const drop = (ev) => {
    ev.preventDefault();
    const data = ev.dataTransfer;
    const fileList = [];

    if (data.items) {
      for (let i = 0; i < data.items.length; i++) {
        if (data.items[i].kind === 'file') {
          const file = data.items[i].getAsFile();
          fileList.push(file);
          displayImage(file);
          logToFile(file);
        }
      }
      sendToServer(fileList);
    }
  };

  const openFileInput = () => {
    const fileInput = document.getElementById('fileInput');
    fileInput.click();
  };

  const handleFileSelect = (event) => {
    const files = event.target.files;
    if (files.length > 0) {
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        displayImage(file);
        logToFile(file);

      }
      sendToServer(files);
    }
  };



  const displayImage = (file) => {
    const reader = new FileReader();

    reader.onload = function (e) {
      const imageContainer = document.getElementById('image-container');
      const imgElement = document.createElement('img');
      imgElement.src = e.target.result;
      imgElement.alt = file.name;
      imgElement.style.maxWidth = '30%';
      imgElement.style.maxHeight = '30%';
      imageContainer.appendChild(imgElement);
    };

    reader.readAsDataURL(file);
  };

  const logToFile = (file) => {
    console.log('Selected file:', file);
  };

  const sendToServer = (files) => {
    const formData = new FormData();

    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }

    // You can use the fetch API to send the file and key to the server
    fetch('http://203.252.161.105/upload', {
      method: 'POST',
      // mode: 'no-cors',
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        console.log('Server response:', data);

        setServerUrl(data['download_url']);
      })
      .catch((error) => console.error('Error sending file to server:', error));
  };


  const retrieveUrl = () => {
    // You can use the fetch API to retrieve the URL based on the entered key
    fetch(`http://203.252.161.105/output/key=${enteredKey}`,)
      .then((response) => response.json())
      .then((data) => {
        console.log('Retrieved URL:', data.url);
        setServerUrl(data.url);
      })
      .catch((error) => console.error('Error retrieving URL from server:', error));
  };

  return (
    <div>
      <div onDrop={drop} onDragOver={dragEnter} style={{ border: '1px solid black', padding: '5rem' }}>
        <p>여기에 파일을 드랍하세요</p>
      </div>
      <div>
        <label htmlFor="fileInput">또는 파일을 선택하세요:</label>
        <input type="file" id="fileInput" style={{ display: 'none' }} multiple onChange={handleFileSelect} />
        <button onClick={openFileInput}>파일 선택</button>
      </div>
      <div id="image-container"></div>
      <div>
        <p>서버로부터 받은 URL: {serverUrl}</p>
      </div>
    </div>
  );
};

export default ImageUploader;
