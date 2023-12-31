let net;


const webcamElement = document.getElementById("webcam");
const classifier = knnClassifier.create();
let modelo;

let webcam;

const video = document.getElementById("webcam");
const constraints = {
  audio:false, 
  video:{
    width:640,
    height:480
  }
}
async function initVideo(){
    try{
      const streamVideo = await navigator.mediaDevices.getUserMedia(constraints)
      handleSucces(streamVideo)
    }
    catch(e){
      alert(e)
    }
  }
  // on succes
  function handleSucces(streamVideo){
    window.stream = streamVideo;
    video.srcObject = streamVideo;
  }



initVideo()


async function app(){
    if ('mediaDevices' in navigator && 'getUserMedia' in navigator.mediaDevices) {
        console.log("Let's get this party started")
    }
    navigator.mediaDevices.getUserMedia({video: true})
    modelo = await fetch("ModeloCemento.txt")
    modelo = await modelo.text();
    // console.log(modelo);
    classifier.setClassifierDataset(Object.fromEntries(JSON.parse(modelo).map(([label, data, shape]) => [label, tf.tensor(data, shape)])));
    net = await mobilenet.load();
    webcam = await tf.data.webcam(webcamElement);
    while(true){
        const img = await webcam.capture();
        const result = await net.classify(img);
        const activation = net.infer(img, "conv_preds");
        let result2;
        try {
            result2 = await classifier.predictClass(activation);
            const classes = ["Undefined","Cemento Extraforte", "Cemento Fortimax"]
            document.getElementById("console2").innerText="Prediction: " + classes[result2.label] + " probability: " + result2.confidences[result2.label]; 
        } catch (error) {
            console.log("Modelo no configurado aún");
            result2= {};
            document.getElementById("console2").innerText="Untrained"                   
        }
        try{
            
        }
        catch(err) {
        }
        img.dispose();
        await tf.nextFrame();
    }


}

async function addExample(classId){
    console.log('Added example');
    const img = await webcam.capture();
    const activation = net.infer(img, true);
    classifier.addExample(activation, classId);
    img.dispose();
}
const saveKnn = async () => {
    let strClassifier = JSON.stringify(Object.entries(classifier.getClassifierDataset()).map(([label, data]) => [label, Array.from(data.dataSync()), data.shape]));
    const storageKey = "knnClassifier";
    localStorage.setItem(storageKey, strClassifier);
};


const loadKnn = async ()=>{
    const storageKey = "knnClassifier";
    let datasetJson = localStorage.getItem(storageKey);
    classifier.setClassifierDataset(Object.fromEntries(JSON.parse(datasetJson).map(([label, data, shape]) => [label, tf.tensor(data, shape)])));
};

app();
