const cv = require('opencv.js');
const fs = require('fs-extra');
const NodeWebcam = require('node-webcam');
const sharp = require('sharp');
const path = require('path');

const { execSync } = require('child_process');

class EmotionDetector {
    constructor() {
        this.faceCascade = null;
        this.emotionLabels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'];
        this.isInitialized = false;
        
        // Webcam configuration for Raspberry Pi
        this.webcamOpts = {
            width: 640,
            height: 480,
            quality: 100,
            delay: 0,
            saveShots: true,
            output: "jpeg",
            device: false,
            callbackReturn: "location",
            verbose: false
        };
        
        this.webcam = NodeWebcam.create(this.webcamOpts);
    }

    async initialize() {
        try {
            console.log('Initializing OpenCV.js...');
            
            // Wait for OpenCV to be ready
            await this.waitForOpenCV();
            
            // Load face cascade
            await this.loadFaceCascade();
            
            console.log('OpenCV.js initialized successfully!');
            this.isInitialized = true;
            
            return true;
        } catch (error) {
            console.error('Failed to initialize:', error);
            return false;
        }
    }

    async waitForOpenCV() {
        return new Promise((resolve) => {
            if (cv.getBuildInformation) {
                console.log('OpenCV.js is ready');
                resolve();
            } else {
                // If OpenCV is not ready, wait a bit and try again
                setTimeout(() => this.waitForOpenCV().then(resolve), 100);
            }
        });
    }

    async loadFaceCascade() {
        try {
            console.log('Loading face cascade for OpenCV.js...');
            
            // For opencv.js, we need to use the built-in cascade or load it differently
            // opencv.js has built-in cascades that we can access
            
            // Method 1: Use built-in cascade data
            this.faceCascade = new cv.CascadeClassifier();
            
            // Load the default frontal face cascade that comes with opencv.js
            // opencv.js includes some pre-compiled cascades
            const cascadeFile = 'haarcascade_frontalface_default.xml';
            
            // For opencv.js, we need to load the cascade data differently
            // Let's try to create a cascade with the built-in data
            try {
                // This is the correct way for opencv.js
                this.faceCascade.load(cascadeFile);
                
                if (this.faceCascade.empty()) {
                    throw new Error('Cascade is empty after loading');
                }
                
                console.log('Face cascade loaded successfully');
                return;
                
            } catch (loadError) {
                console.log('Built-in cascade loading failed, trying alternative method...');
                
                // Alternative: Load from file system if available
                await this.loadCascadeFromFile();
            }
            
        } catch (error) {
            console.error('Error loading face cascade:', error);
            throw new Error('Failed to load face cascade');
        }
    }

    async loadCascadeFromFile() {
        const cascadePath = path.join(__dirname, 'models', 'haarcascade_frontalface_default.xml');
        
        try {
            // Ensure we have the cascade file
            if (!await fs.pathExists(cascadePath)) {
                await this.downloadHaarCascade(cascadePath);
            }
            
            // For opencv.js, we might need to read the file and process it differently
            const cascadeData = await fs.readFile(cascadePath, 'utf8');
            
            // opencv.js expects cascade data in a specific format
            // We'll use a simpler approach with manual face detection
            console.log('Using alternative face detection method...');
            this.faceCascade = null; // We'll implement manual detection
            
        } catch (error) {
            console.error('Failed to load cascade from file:', error);
            throw error;
        }
    }

    async downloadHaarCascade(cascadePath) {
        try {
            console.log('Downloading Haar cascade file...');
            await fs.ensureDir(path.dirname(cascadePath));
            
            const { exec } = require('child_process');
            const util = require('util');
            const execPromise = util.promisify(exec);
            
            const url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml';
            
            await execPromise(`curl -o "${cascadePath}" "${url}"`);
            
            if (!await fs.pathExists(cascadePath)) {
                throw new Error('Failed to download cascade file');
            }
            
            console.log('Haar cascade downloaded successfully');
            
        } catch (error) {
            console.error('Download failed:', error);
            throw error;
        }
    }

    // Capture image using libcamera-still
    async captureImage(outputPath = 'capture.jpg') {
        try {
            const command = `libcamera-still -o ${outputPath} --nopreview --timeout 1000 --width 640 --height 480`;
            await execSync(command);
            console.log(`Image captured: ${outputPath}`);
            return outputPath;
        } catch (error) {
            console.error('Error capturing image:', error.message);
            throw error;
        }
    }

    async processImage(imagePath) {
        let src = null;
        let gray = null;
        
        try {
            console.log('Processing image:', imagePath);
            
            // Verify image exists
            if (!await fs.pathExists(imagePath)) {
                throw new Error(`Image file not found: ${imagePath}`);
            }
            
            // Read image with Sharp and convert to format opencv.js can handle
            const imageBuffer = await sharp(imagePath)
                .ensureAlpha()
                .raw()
                .toBuffer({ resolveWithObject: true });
            
            const { data, info } = imageBuffer;
            
            console.log('Image loaded:', info);
            
            // Create Mat from image data
            src = cv.matFromImageData({
                data: new Uint8ClampedArray(data),
                width: info.width,
                height: info.height
            });
            
            if (src.empty()) {
                throw new Error('Failed to create source image matrix');
            }
            
            // Convert to grayscale
            gray = new cv.Mat();
            cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
            
            if (gray.empty()) {
                throw new Error('Failed to convert to grayscale');
            }
            
            console.log('Image converted to grayscale');
            
            // Detect faces
            const faces = await this.detectFaces(gray);
            
            const results = [];
            
            // Process detected faces
            faces.forEach((face, index) => {
                console.log(`Processing face ${index + 1}:`, face);
                
                const emotion = this.predictEmotion();
                
                results.push({
                    x: face.x,
                    y: face.y,
                    width: face.width,
                    height: face.height,
                    emotion: emotion
                });
            });
            
            return results;
            
        } catch (error) {
            console.error('Error processing image:', error);
            return [];
        } finally {
            // Clean up
            if (src) src.delete();
            if (gray) gray.delete();
        }
    }

    async detectFaces(grayMat) {
        try {
            if (this.faceCascade && !this.faceCascade.empty()) {
                // Use Haar cascade if available
                const faces = new cv.RectVector();
                
                this.faceCascade.detectMultiScale(
                    grayMat,
                    faces,
                    1.1,    // scale factor
                    3,      // min neighbors
                    0,      // flags
                    new cv.Size(30, 30) // min size
                );
                
                const results = [];
                for (let i = 0; i < faces.size(); i++) {
                    const face = faces.get(i);
                    results.push({
                        x: face.x,
                        y: face.y,
                        width: face.width,
                        height: face.height
                    });
                }
                
                faces.delete();
                return results;
                
            } else {
                // Fallback: Use simple blob detection or contour detection
                console.log('Using alternative face detection...');
                return await this.alternativeFaceDetection(grayMat);
            }
            
        } catch (error) {
            console.error('Face detection error:', error);
            // Return empty array if detection fails
            return [];
        }
    }

    async alternativeFaceDetection(grayMat) {
        try {
            // Simple alternative: look for face-like regions using contours
            const thresh = new cv.Mat();
            cv.threshold(grayMat, thresh, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
            
            const contours = new cv.MatVector();
            const hierarchy = new cv.Mat();
            
            cv.findContours(thresh, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
            
            const faces = [];
            
            // Look for contours that might be faces (rough approximation)
            for (let i = 0; i < contours.size(); i++) {
                const contour = contours.get(i);
                const rect = cv.boundingRect(contour);
                
                // Filter by size (faces should be reasonably sized)
                if (rect.width > 50 && rect.height > 50 && 
                    rect.width < grayMat.cols * 0.8 && rect.height < grayMat.rows * 0.8) {
                    
                    // Check aspect ratio (faces are roughly square-ish)
                    const aspectRatio = rect.width / rect.height;
                    if (aspectRatio > 0.6 && aspectRatio < 1.4) {
                        faces.push(rect);
                    }
                }
                
                contour.delete();
            }
            
            // Clean up
            thresh.delete();
            contours.delete();
            hierarchy.delete();
            
            return faces;
            
        } catch (error) {
            console.error('Alternative detection error:', error);
            return [];
        }
    }

    predictEmotion() {
        // Mock emotion prediction
        const emotions = ['Happy', 'Sad', 'Angry', 'Surprise', 'Fear', 'Disgust', 'Neutral'];
        return emotions[Math.floor(Math.random() * emotions.length)];
    }

    async detectEmotions() {
        if (!this.isInitialized) {
            console.error('Detector not initialized');
            return [];
        }

        let imagePath = null;
        
        try {
            console.log('Capturing image...');
            imagePath = await this.captureImage();
            console.log('Image captured:', imagePath);
            
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
            
            return results;
            
        } catch (error) {
            console.error('Error detecting emotions:', error);
            return [];
        } finally {
            // Clean up
            if (imagePath && await fs.pathExists(imagePath)) {
                try {
                    // await fs.remove(imagePath);
                } catch (cleanupError) {
                    console.warn('Failed to clean up image:', cleanupError.message);
                }
            }
        }
    }

    async startRealTimeDetection(intervalMs = 3000) {
        console.log('Starting real-time emotion detection...');
        console.log('Press Ctrl+C to stop');
        
        const detect = async () => {
            const results = await this.detectEmotions();
            setTimeout(detect, intervalMs);
        };
        
        detect();
    }
}

// Main execution
async function main() {
    const detector = new EmotionDetector();
    
    console.log('Initializing emotion detector...');
    const initialized = await detector.initialize();
    
    if (!initialized) {
        console.error('Failed to initialize emotion detector');
        process.exit(1);
    }
    
    // Run detection
    if (process.argv.includes('--realtime')) {
        await detector.startRealTimeDetection(3000);
    } else {
        console.log('Running single detection...');
        const results = await detector.detectEmotions();
        console.log('Detection complete.');
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