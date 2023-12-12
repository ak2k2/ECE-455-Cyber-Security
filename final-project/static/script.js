var socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port);

function startVideo() {
    var video = document.getElementById('video');
    var canvasOverlay = document.getElementById('videoCanvasOverlay');
    var contextOverlay = canvasOverlay.getContext('2d');

    // Get user media
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
            video.onloadedmetadata = function (e) {
                canvasOverlay.width = video.videoWidth;
                canvasOverlay.height = video.videoHeight;
                video.play();
            };
        })
        .catch(err => {
            console.log("An error occurred: " + err);
        });

    // Send frame to server
    video.addEventListener('play', () => {
        function sendFrame() {
            if (video.paused || video.ended) {
                return;
            }
            socket.emit('stream_frame', getFrame(video));
            setTimeout(sendFrame, 50); // Adjust this based on performance
        }
        sendFrame();
    });
}

function getFrame(video) {
    var canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    var ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg');
}

// Processsed frame from server
socket.on('processed_frame', function (data) {
    var image = new Image();
    image.onload = function () {
        var canvasOverlay = document.getElementById('videoCanvasOverlay');
        var contextOverlay = canvasOverlay.getContext('2d');
        contextOverlay.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);
        contextOverlay.drawImage(image, 0, 0, canvasOverlay.width, canvasOverlay.height);
    };
    image.src = 'data:image/jpeg;base64,' + data;
});


window.onload = startVideo;
