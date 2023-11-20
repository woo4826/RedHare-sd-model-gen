// src/ImageUploader.js

import React, { useState } from 'react';

const ImageUploader = () => {
  const [fileKey, setFileKey] = useState(null);
  const [serverUrl, setServerUrl] = useState(null);
  const [enteredKey, setEnteredKey] = useState('');

  const dragEnter = (ev) => {
    ev.preventDefault();
  };

  const drop = (ev) => {
    ev.preventDefault();
    const data = ev.dataTransfer;
    if (data.items) {
      for (let i = 0; i < data.items.length; i++) {
        if (data.items[i].kind === 'file') {
          const file = data.items[i].getAsFile();
          const key = generateRandomKey();
          setFileKey(key);
          displayImage(file, key);
          logToFile(file, key);
          sendToServer(file, key);
        }
      }
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
        const key = generateRandomKey();
        setFileKey(key);
        displayImage(file, key);
        logToFile(file, key);
        sendToServer(file, key);
      }
    }
  };

  const generateRandomKey = () => {
    return Math.random().toString(36).substring(7);
  };

  const displayImage = (file, key) => {
    const reader = new FileReader();

    reader.onload = function (e) {
      const imageContainer = document.getElementById('image-container');
      const imgElement = document.createElement('img');
      imgElement.src = e.target.result;
      imgElement.alt = file.name;
      imgElement.style.maxWidth = '30%';
      imgElement.style.maxHeight = '30%';
      imageContainer.innerHTML = `<p>File Key: ${key}</p>`;
      imageContainer.appendChild(imgElement);
    };

    reader.readAsDataURL(file);
  };

  const logToFile = (file, key) => {
    console.log('Selected file:', file);
    console.log('File Key:', key);
  };

  const sendToServer = (file, key) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('key', key);

    // You can use the fetch API to send the file and key to the server
    fetch('5000/upload', {
      method: 'POST',
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        console.log('Server response:', data);
        setServerUrl(data.url);
      })
      .catch((error) => console.error('Error sending file to server:', error));
  };

  const handleKeyInputChange = (event) => {
    setEnteredKey(event.target.value);
  };

  const retrieveUrl = () => {
    // You can use the fetch API to retrieve the URL based on the entered key
    fetch(`5000/outpus/key=${enteredKey}`)
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
        <label htmlFor="enteredKey">URL을 가져오려면 키를 입력하세요:</label>
        <input type="text" id="enteredKey" value={enteredKey} onChange={handleKeyInputChange} />
        <button onClick={retrieveUrl}>URL 가져오기</button>
      </div>
    </div>
  );
};

export default ImageUploader;
