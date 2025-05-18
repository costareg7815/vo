document.getElementById('createVideosForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const formData = new FormData();
    const imageFile = document.getElementById('image').files[0];
    const videoCount = document.getElementById('videoCount').value;
    const voice = document.getElementById('voice').value;
    const text = document.getElementById('text').value;

    // إضافة البيانات إلى FormData
    formData.append('image', imageFile);
    formData.append('videoCount', videoCount);
    formData.append('voice', voice);
    formData.append('text', text);

    fetch('/create_videos', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const videosContainer = document.getElementById('videos');
        videosContainer.innerHTML = '';
        
        data.forEach(video => {
            const videoElement = document.createElement('video');
            videoElement.src = video.url;
            videoElement.controls = true;
            videoElement.className = 'video';
            videosContainer.appendChild(videoElement);
        });

        alert("🎉 Video(s) created successfully!");
    })
    .catch(error => console.error('❌ Error:', error));
});
