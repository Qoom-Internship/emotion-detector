const cv = require('opencv.js');
const fs = require('fs-extra');
const path = require('path');
// const NodeWebcam = require('node-webcam');
const sharp = require('sharp');
const { execSync } = require('child_process');

class EmotionDetector {
    constructor() {
        this.faceCascade = null;
        this.emotionLabels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'];
        this.isInitialized = false;
        
        // // Webcam configuration for Raspberry Pi
        // this.webcamOpts = {
        //     width: 640,
        //     height: 480,
        //     quality: 100,
        //     delay: 0,
        //     saveShots: true,
        //     output: "jpeg",
        //     device: false, // Use default camera
        //     callbackReturn: "location",
        //     verbose: false
        // };
        
        // this.webcam = NodeWebcam.create(this.webcamOpts);
    }

    async initialize() {
        try {
            console.log('Initializing OpenCV...');
            
            // Load Haar cascade for face detection
            await this.loadHaarCascade();
            
            console.log('OpenCV initialized successfully!');
            this.isInitialized = true;
            
            return true;
        } catch (error) {
            console.error('Failed to initialize:', error);
            return false;
        }
    }

    async loadHaarCascade() {
        // You need to download the haarcascade file
        const cascadePath = './models/haarcascade_frontalface_default.xml';
        
        if (!await fs.pathExists(cascadePath)) {
            console.log('Downloading Haar cascade file...');
            await this.downloadHaarCascade();
        }
        
        this.faceCascade = new cv.CascadeClassifier();
        this.faceCascade.load(cascadePath);
    }

    async downloadHaarCascade() {
        // Create models directory
        await fs.ensureDir('./models');
        
        // For now, you'll need to manually download this file
        // from: https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
        console.log('Please download haarcascade_frontalface_default.xml to ./models/ directory');
        console.log('URL: https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml');
        throw new Error('Haar cascade file not found. Please download manually.');
    }

    // captureImage() {
    //     return new Promise((resolve, reject) => {
    //         this.webcam.capture("capture", (err, data) => {
    //             if (err) {
    //                 reject(err);
    //             } else {
    //                 resolve(data);
    //             }
    //         });
    //     });
    // }
    async captureImage() {
        try {
            console.log('Capturing image from Pi Camera...');

            const timestamp = Date.now();
            const filename = `pi_camera_capture_${timestamp}.jpg`;
            const filePath = path.resolve(__dirname, filename);

            // 2초간 카메라 준비 후 촬영
            execSync(`libcamera-still -t 2000 -o ${filePath} --nopreview`, { stdio: 'ignore' });

            console.log(`Image captured: ${filePath}`);
            return filePath;
        } catch (error) {
            throw new Error(`Failed to capture image with Pi Camera: ${error.message}`);
        }
    }

    async processImage(imagePath) {
        try {
            // Read image using Sharp and convert to buffer
            const imageBuffer = await sharp(imagePath)
                .raw()
                .toBuffer({ resolveWithObject: true });
            
            // Convert to OpenCV Mat
            const { data, info } = imageBuffer;
            const mat = cv.matFromImageData({
                data: new Uint8ClampedArray(data),
                width: info.width,
                height: info.height
            });

            // Convert to grayscale for face detection
            const gray = new cv.Mat();
            cv.cvtColor(mat, gray, cv.COLOR_RGBA2GRAY);

            // Detect faces
            const faces = new cv.RectVector();
            this.faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0);

            const results = [];
            
            // Process each detected face
            for (let i = 0; i < faces.size(); i++) {
                const face = faces.get(i);
                
                // Extract face region
                const faceROI = gray.roi(face);
                
                // Resize face for emotion detection (typically 48x48 for emotion models)
                const resizedFace = new cv.Mat();
                cv.resize(faceROI, resizedFace, new cv.Size(48, 48));
                
                // For now, we'll use a simple mock emotion prediction
                // In a real implementation, you'd use a trained model here
                const emotion = this.predictEmotion(resizedFace);
                
                results.push({
                    x: face.x,
                    y: face.y,
                    width: face.width,
                    height: face.height,
                    emotion: emotion
                });

                // Clean up
                faceROI.delete();
                resizedFace.delete();
            }

            // Clean up
            mat.delete();
            gray.delete();
            faces.delete();

            return results;
            
        } catch (error) {
            console.error('Error processing image:', error);
            return [];
        }
    }

    // Mock emotion prediction - replace with actual ML model
    predictEmotion(faceROI) {
        // This is a placeholder. For real emotion detection, you'd need:
        // 1. A trained emotion recognition model (like FER2013)
        // 2. TensorFlow.js or similar to run the model
        // 3. Proper preprocessing of the face image
        
        // For now, return a random emotion for demonstration
        const emotions = ['Happy', 'Sad', 'Angry', 'Surprise', 'Fear', 'Disgust', 'Neutral'];
        return emotions[Math.floor(Math.random() * emotions.length)];
    }

    async detectEmotions() {
        if (!this.isInitialized) {
            console.error('Detector not initialized');
            return;
        }

        try {
            console.log('Capturing image...');
            const imagePath = await this.captureImage();
            
            console.log('Processing image for emotions...');
            const results = await this.processImage(imagePath);
            
            if (results.length > 0) {
                console.log('Detected faces and emotions:');
                results.forEach((result, index) => {
                    console.log(`Face ${index + 1}: ${result.emotion} at (${result.x}, ${result.y})`);
                });
            } else {
                console.log('No faces detected');
            }
            
            // Clean up captured image
            await fs.remove(imagePath);
            
            return results;
            
        } catch (error) {
            console.error('Error detecting emotions:', error);
            return [];
        }
    }

    async startRealTimeDetection(intervalMs = 2000) {
        console.log('Starting real-time emotion detection...');
        console.log('Press Ctrl+C to stop');
        
        const detect = async () => {
            await this.detectEmotions();
            setTimeout(detect, intervalMs);
        };
        
        detect();
    }
}

// Main execution
async function main() {
    const detector = new EmotionDetector();
    
    const initialized = await detector.initialize();
    if (!initialized) {
        console.error('Failed to initialize emotion detector');
        process.exit(1);
    }
    
    // You can either detect once or start real-time detection
    if (process.argv.includes('--realtime')) {
        await detector.startRealTimeDetection(3000); // Every 3 seconds
    } else {
        // Single detection
        const results = await detector.detectEmotions();
        console.log('Detection complete:', results);
    }
}

// Handle graceful shutdown
process.on('SIGINT', () => {
    console.log('\nShutting down gracefully...');
    process.exit(0);
});

// Run the application
if (require.main === module) {
    main().catch(console.error);
}

module.exports = EmotionDetector;