import UIKit
import AVFoundation
import Vision

class CameraViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    var captureSession: AVCaptureSession!
    var previewLayer: AVCaptureVideoPreviewLayer!
    var shouldProcessFrame = true

    // Core ML Model
    private var model: VNCoreMLModel!

    override func viewDidLoad() {
        super.viewDidLoad()
        setupModel()
        setupCamera()
    }

    // MARK: - Setup Core ML Model
    private func setupModel() {
        guard let mlModel = try? BallDetector(configuration: MLModelConfiguration()).model else {
            fatalError("Could not load BallDetector model")
        }
        model = try? VNCoreMLModel(for: mlModel)
    }

    // MARK: - Setup Camera
    private func setupCamera() {
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .high

        guard let videoCaptureDevice = AVCaptureDevice.default(for: .video) else {
            fatalError("No camera available")
        }

        let videoInput: AVCaptureDeviceInput
        do {
            videoInput = try AVCaptureDeviceInput(device: videoCaptureDevice)
        } catch {
            fatalError("Error accessing camera: \(error)")
        }

        if captureSession.canAddInput(videoInput) {
            captureSession.addInput(videoInput)
        } else {
            fatalError("Could not add camera input")
        }

        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "camera.queue"))

        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        } else {
            fatalError("Could not add video output")
        }

        // Disable mirroring if needed
        if let connection = videoOutput.connection(with: .video), connection.isVideoMirroringSupported {
            connection.isVideoMirrored = false
        }

        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.videoGravity = .resizeAspect
        previewLayer.frame = view.bounds
        view.layer.addSublayer(previewLayer)

        DispatchQueue.global(qos: .userInitiated).async {
            self.captureSession.startRunning()
        }
    }

    // MARK: - Process Frames
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }

        if shouldProcessFrame {
            shouldProcessFrame = false

            // Run the Vision request on a background thread
            DispatchQueue.global(qos: .userInitiated).async {
                self.detectBall(in: pixelBuffer)

                // Throttle to 10 FPS
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                    self.shouldProcessFrame = true
                }
            }
        }
    }

    // MARK: - Detect Ball
    private func detectBall(in pixelBuffer: CVPixelBuffer) {
        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            guard let results = request.results as? [VNRecognizedObjectObservation] else {
                return
            }

            // Process the detection results
            self?.handleDetections(results)
        }

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        do {
            try handler.perform([request])
        } catch {
            print("Failed to perform request: \(error)")
        }
    }

    // MARK: - Handle Detections
    private func handleDetections(_ detections: [VNRecognizedObjectObservation]) {
        DispatchQueue.main.async {
            // Clear old bounding boxes
            self.view.layer.sublayers?.removeSubrange(1...)

            // Draw bounding boxes for all detections
            for detection in detections {
                let boundingBox = detection.boundingBox
                let box = self.createBoundingBox(from: boundingBox)
                self.view.layer.addSublayer(box)
            }
        }
    }

    // MARK: - Create Bounding Box
    private func createBoundingBox(from rect: CGRect) -> CAShapeLayer {
        let layer = CAShapeLayer()

        // Convert normalized Vision coordinates to screen coordinates
        let convertedRect = convertBoundingBox(rect: rect)
        layer.frame = convertedRect
        layer.borderColor = UIColor.red.cgColor
        layer.borderWidth = 2
        return layer
    }

    private func convertBoundingBox(rect: CGRect) -> CGRect {
        // Vision coordinates are normalized with (0,0) at bottom-left
        // UIKit coordinates have (0,0) at top-left
        let x = rect.origin.y * view.bounds.width   // Swap x with y
        let y =  rect.origin.x * view.bounds.height // Swap y with x and invert
        let width = rect.height * view.bounds.width // Swap width with height
        let height = rect.width * view.bounds.height // Swap height with width

        return CGRect(x: x, y: y, width: width, height: height)
    }
}
