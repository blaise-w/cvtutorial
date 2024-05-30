let video;
let poseNet;
let pose;
let skeleton;
let brain;
let state = 'waiting';
let poseLabel = '';
let poseCount = 2;


function setup() {
    let canvas = createCanvas(640, 480);
    canvas.parent('sketch-holder');

    video = createCapture(VIDEO);
    video.size(width, height);
    video.hide();

    poseNet = ml5.poseNet(video, modelReady);
    poseNet.on('pose', gotPoses);

    let options = {
        inputs: 34,
        outputs: 4,
        task: 'classification',
        debug: true
    };
    brain = ml5.neuralNetwork(options);

    setupButtons();
}

function modelReady() {
    console.log('PoseNet is ready.');
}

function gotPoses(results) {
    if (results.length > 0) {
        pose = results[0].pose;
        skeleton = results[0].skeleton;
        if (state === 'collecting') {
            let inputs = pose.keypoints.map(p => [p.position.x, p.position.y]).flat();
            brain.addData(inputs, [poseLabel]);
        }
    }
}

function draw() {
    push();
    translate(video.width, 0);
    scale(-1, 1);
    image(video, 0, 0, width, height);
    if (pose) {
        drawKeypoints();
        drawSkeleton();
    }
    pop();

    fill(255, 0, 255);
    noStroke();
    textSize(64);
    textAlign(CENTER, CENTER);
    text(poseLabel, width / 2, height / 2);
}

function drawKeypoints() {
    if (pose) {
        pose.keypoints.forEach(keypoint => {
            if (keypoint.score > 0.2) {
                fill(0, 255, 0);
                noStroke();
                ellipse(keypoint.position.x, keypoint.position.y, 16, 16);
            }
        });
    }
}

function drawSkeleton() {
    if (skeleton) {
        skeleton.forEach(link => {
            let [a, b] = link;
            stroke(255, 0, 0);
            strokeWeight(2);
            line(a.position.x, a.position.y, b.position.x, b.position.y);
        });
    }
}


function setupButtons() {
    document.getElementById('addPose').addEventListener('click', addPose);
    document.getElementById('removePose').addEventListener('click', removePose);
    document.getElementById('trainModel').addEventListener('click', trainModel);
    document.getElementById('startCollecting1').addEventListener('click', () => startCollecting(1));
    document.getElementById('startCollecting2').addEventListener('click', () => startCollecting(2));
}

function startCollecting(poseNumber) {
    const poseInput = document.getElementById(`pose${poseNumber}`);
    poseLabel = poseInput.value;
    state = 'collecting';
    console.log(`Starting data collection for label: ${poseLabel}`);
    setTimeout(() => {
        state = 'waiting';
        console.log(`Stopped collecting data for label: ${poseLabel}`);
    }, 10000); // Collect data for 10 seconds
}

function addPose() {
    if (poseCount < 4) {
        poseCount++;
        const div = document.createElement('div');
        div.className = 'poseInput';
        div.innerHTML = `
            <input type="text" id="pose${poseCount}" placeholder="Enter label for Pose ${poseCount}">
            <button onclick="startCollecting(${poseCount})">Record Pose ${poseCount}</button>
        `;
        document.getElementById('poseInputs').appendChild(div);
    } else {
        alert('You can only add up to four poses.');
    }
}

function removePose() {
    if (poseCount > 2) {
        const poseInputs = document.getElementById('poseInputs');
        poseInputs.removeChild(poseInputs.lastChild);
        poseCount--;
    } else {
        alert('You must have at least two poses.');
    }
}

function trainModel() {
    console.log('Starting model training...');
    brain.normalizeData();
    brain.train({epochs: 50}, () => {
        console.log('Model trained.');
        classifyPose();
    });
}

function classifyPose() {
    if (pose) {
        let inputs = pose.keypoints.map(p => [p.position.x, p.position.y]).flat();
        brain.classify(inputs, gotResult);
    } else {
        setTimeout(classifyPose, 100);
    }
}

function gotResult(error, results) {
    if (results && results.length > 0 && results[0].confidence > 0.75) {
        poseLabel = results[0].label.toUpperCase();
        console.log(`Pose classified as ${poseLabel} with confidence ${results[0].confidence}`);
    }
    setTimeout(classifyPose, 100);
}
