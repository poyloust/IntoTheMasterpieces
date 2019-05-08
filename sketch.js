let video;
const knnClassifier = ml5.KNNClassifier();
let poseNet;
let poses = [];

var rabbit = false;
var pig = false;
var sheep = false;

var currentResult = 0;
var pResult;
let style1;
let style2;
let style3;
let resultImg;

var artwork = document.getElementById('resultImage');
var artist = document.getElementById('artist');
var artTitle = document.getElementById('artworkTitle');
var introtext = document.getElementById('resultText');
var btn = document.getElementById('buttonPredict');
function setup() {

  var canvas = createCanvas(580, 800);
  canvas.parent('videoContainer');
  video = createCapture(VIDEO);
  video.hide();
  video.size(1280, 960);
  //resizeCanvas(0.5 * windowWidth, 0.9 * windowHeight);
  createButtons();
  poseNet = ml5.poseNet(video, modelReady);
  poseNet.on('pose', function (results) {
    poses = results;
  });
  video.hide();

  resultImg = createImg('');
  resultImg.hide();
  style1 = ml5.styleTransfer('models/picassoModel', video, modelALoaded);
  style2 = ml5.styleTransfer('models/vangoghModel', video, modelBLoaded);
  style3 = ml5.styleTransfer('models/screamModel', video, modelCLoaded);
  loadMyKNN();
}

function draw() {
  if (currentResult == 0) {
    //console.log('show original vid');
    image(video, -300, 0, 1280, 960);
    // drawFace();
    // drawKeypoints();
    // drawSkeleton();
  }
}

//////////
//////////

function modelReady() {
  console.log('pose net model loaded');
  btn.style.display = "inline-block";
}
function modelALoaded() {
  console.log('style transfer picassoModel model loaded');
}
function modelBLoaded() {
  console.log('style transfer vangoghModel loaded');
}
function modelCLoaded() {
  console.log('style transfer popartModel loaded');
}

///////
///////
function landing(){
  var welcome = document.getElementById('welcome')
  welcome.style.display = 'none';
}
function classify() {
  const numClasses = knnClassifier.getNumLabels();
  if (numClasses <= 0) {
    console.error('There is no examples in any class');
    return;
  }
  // Convert poses results to a 2d array [[score0, x0, y0],...,[score16, x16, y16]]
  const poseArray = poses[0].pose.keypoints.map(p => [p.score, p.position.x, p.position.y]);

  // Create a tensor2d from 2d array
  const logits = ml5.tf.tensor2d(poseArray);

  // Use knnClassifier to classify which class do these features belong to
  // You can pass in a callback function `poseResults` to knnClassifier.classify function
  knnClassifier.classify(logits, poseResults);
}

/// this function need further optimise

function poseResults(err, result) {
  if (err) {
    console.error(err);
    classify();
  }

  if (result.confidencesByLabel) {
    const confideces = result.confidencesByLabel;
    // result.label is the label that has the highest confidence
    if (result.label) {
      select('#result').html(result.label);
      select('#confidence').html(`${confideces[result.label] * 100} %`);
    }

    if (confideces['A'] > 0.9) {
      rabbit = true;
      currentResult = 1;
    }
    else {
      rabbit = false;
    }
    if (confideces['B'] > 0.9) {
      pig = true;
      currentResult = 2;
    }
    else {
      pig = false;
    }
    if (confideces['C'] > 0.9) {
      sheep = true;
      currentResult = 3;
    }
    else {
      sheep = false;
    }

    if (pResult != currentResult) {
      console.log('change of state, now is ' + currentResult);
      if (currentResult == 1) {
        style1.transfer(style1Result);
        artist.innerText = "Publo Picasso";
        artTitle.innerText = "Le Rêve";  
        introtext.innerText = "Le Rêve (French, \"The Dream\") is a 1932 oil painting (130 × 97 cm) by Pablo Picasso, then 50 years old, portraying his 22-year-old mistress Marie-Thérèse Walter. It is said to have been painted in one afternoon, on 24 January 1932. It belongs to Picasso's period of distorted depictions, with its oversimplified outlines and contrasted colors resembling early Fauvism.";
        artwork.style.backgroundImage = "url('images/picasso.jpg')";
        artwork.style.width = "220px";
        artwork.style.marginBottom = "40px"
      }
      if (currentResult == 2){
        style2.transfer(style2Result);
        artist.innerText = "Vincent van Gogh";
        artTitle.innerText = "The Siesta";  
        introtext.innerText = "The siesta was painted while Van Gogh was interned in a mental asylum in Saint-Rémy de Provence. Van Gogh uses color to depict the peaceful nature of the mid-day rest. Use of contrasting colors, blue-violet against yellow-orange brings an intensity to the work that is uniquely his style.";
        artwork.style.backgroundImage = "url('images/vangogh.jpg')";
        artwork.style.removeProperty('width');
        artwork.style.marginBottom = "60px"
      }
      if (currentResult == 3){
        style3.transfer(style3Result);
        artist.innerText = "Edward Munch";
        artTitle.innerText = "The Scream of Nature";  
        introtext.innerText = "The agonised face in the painting has become one of the most iconic images of art, seen as symbolising the anxiety of modern man. Munch recalled that he had been out for a walk at sunset when suddenly the setting sunlight turned the clouds \"a blood red\". ";
        artwork.style.backgroundImage = "url('images/scream.webp')";
        artwork.style.width = "220px";
        artwork.style.marginBottom = "50px"
      

      }
    }
    //console.log (pResult, currentResult);
    select('#confidenceA').html(`${confideces['A'] ? confideces['A'] * 100 : 0} %`);
    select('#confidenceB').html(`${confideces['B'] ? confideces['B'] * 100 : 0} %`);
    select('#confidenceC').html(`${confideces['C'] ? confideces['C'] * 100 : 0} %`);
  }
  pResult = currentResult;
  classify();
}

/////////////
/////////////


function style1Result(err, img) {
  resultImg.attribute('src', img.src);
  image(resultImg, -300, 0, 1280, 960);


  if(currentResult==1){
    style1.transfer(style1Result);
  }
}

function style2Result(err, img) {
  resultImg.attribute('src', img.src);
  image(resultImg, -300, 0, 1280, 960);
  if(currentResult==2){
    style2.transfer(style2Result);
  }
}

function style3Result(err, img) {
  resultImg.attribute('src', img.src);
  image(resultImg, -300, 0, 1280, 960);
  if(currentResult==3){
    style3.transfer(style3Result);
  }
}

function addExample(label) {
  // Convert poses results to a 2d array [[score0, x0, y0],...,[score16, x16, y16]]
  const poseArray = poses[0].pose.keypoints.map(p => [p.score, p.position.x, p.position.y]);
  // Create a tensor2d from 2d array
  const logits = ml5.tf.tensor2d(poseArray);
  // Add an example with a label to the classifier
  knnClassifier.addExample(logits, label);
  updateExampleCounts();
}

function createButtons() {
  buttonA = select('#addClassA');
  buttonA.mousePressed(function () {
    addExample('A');
  });
  buttonB = select('#addClassB');
  buttonB.mousePressed(function () {
    addExample('B');
  });
  buttonC = select('#addClassC');
  buttonC.mousePressed(function () {
    addExample('C');
  });
  resetBtnA = select('#resetA');
  resetBtnA.mousePressed(function () {
    clearClass('A');
  });

  resetBtnB = select('#resetB');
  resetBtnB.mousePressed(function () {
    clearClass('B');
  });
  resetBtnB = select('#resetC');
  resetBtnB.mousePressed(function () {
    clearClass('C');
  });

  buttonPredict = select('#buttonPredict');
  buttonPredict.mousePressed(classify);

  // Clear all classes button
  buttonClearAll = select('#clearAll');
  buttonClearAll.mousePressed(clearAllClasses);

  //Save KNN button
  buttonSave = select('#save');
  buttonSave.mousePressed(saveMyKNN);

  // //Load KNN button
  // buttonLoad = select('#load');
  // buttonLoad.mousePressed(loadMyKNN);
}

function updateExampleCounts() {
  const counts = knnClassifier.getCountByLabel();
  select('#exampleA').html(counts['A'] || 0);
  select('#exampleB').html(counts['B'] || 0);
  select('#exampleC').html(counts['C'] || 0);
}
// Clear the examples in one class
function clearClass(classLabel) {
  knnClassifier.clearClass(classLabel);
  updateExampleCounts();
}
// Clear all the examples in all classes
function clearAllClasses() {
  knnClassifier.clearAllClasses();
  updateExampleCounts();
}

// Save dataset as myKNNDataset.json
function saveMyKNN() {
  knnClassifier.save('myKNNDataset');
}

// Load dataset to the classifier
function loadMyKNN() {
  //knnClassifier.load('./myKNNDataset.json', updateCounts);
  knnClassifier.load('./myKNNDataset.json',updateExampleCounts);
}


function drawFace() {
  strokeWeight(2);
  if (poses.length > 0) {
    let pose = poses[0].pose.keypoints;
    // Create a pink ellipse for the nose
    fill(213, 0, 143);;
    let nose = pose[0].position;
    let r_eye = pose[1].position;
    let l_eye = pose[2].position;
    let r_ear = pose[3].position;
    let l_ear = pose[4].position;
  }
}
function drawKeypoints() {
  // Loop through all the poses detected
  for (let i = 0; i < poses.length; i++) {
    // For each pose detected, loop through all the keypoints
    let pose = poses[i].pose;
    for (let j = 0; j < pose.keypoints.length; j++) {
      // A keypoint is an object describing a body part (like rightArm or leftShoulder)
      let keypoint = pose.keypoints[j];
      // Only draw an ellipse is the pose probability is bigger than 0.2
      if (keypoint.score > 0.2) {
        fill(255, 0, 0);
        noStroke();
        ellipse(keypoint.position.x, keypoint.position.y, 10, 10);
      }
    }
  }
}

function drawSkeleton() {
  // Loop through all the skeletons detected
  for (let i = 0; i < poses.length; i++) {
    let skeleton = poses[i].skeleton;
    // For every skeleton, loop through all body connections
    for (let j = 0; j < skeleton.length; j++) {
      let partA = skeleton[j][0];
      let partB = skeleton[j][1];
      stroke(255, 0, 0);
      line(partA.position.x, partA.position.y, partB.position.x, partB.position.y);
    }
  }
}