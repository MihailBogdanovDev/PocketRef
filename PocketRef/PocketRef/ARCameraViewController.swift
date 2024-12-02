import UIKit
import ARKit
import Vision

class ARCameraViewController: UIViewController, ARSCNViewDelegate, ARSessionDelegate {
    private var sceneView: ARSCNView!
    private var goalLines: [SCNNode] = []
    private var scoreLabel: UILabel!
    private var score: Int = 0

    // Core ML Model
    private var visionModel: VNCoreMLModel!
    private var isProcessingFrame = false

    override func viewDidLoad() {
        super.viewDidLoad()

        // Setup AR Scene
        setupARScene()

        // Setup Score Label
        setupScoreLabel()

        // Setup Core ML Model
        setupModel()

        // Add gesture recognizer for placing goal lines
        let tapGesture = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
        sceneView.addGestureRecognizer(tapGesture)
    }

    private func setupARScene() {
        sceneView = ARSCNView(frame: view.bounds)
        sceneView.delegate = self
        sceneView.session.delegate = self
        sceneView.scene = SCNScene()
        //sceneView.debugOptions = [.showFeaturePoints, .showWorldOrigin] // Debugging options
        view.addSubview(sceneView)

        let configuration = ARWorldTrackingConfiguration()
        configuration.planeDetection = [.horizontal] // Detect horizontal planes
        configuration.isLightEstimationEnabled = true // Enable light estimation
        sceneView.session.run(configuration)
    }

    private func setupScoreLabel() {
        scoreLabel = UILabel()
        scoreLabel.translatesAutoresizingMaskIntoConstraints = false
        scoreLabel.text = "Score: \(score)"
        scoreLabel.font = UIFont.boldSystemFont(ofSize: 24)
        scoreLabel.textColor = .white
        scoreLabel.backgroundColor = UIColor.black.withAlphaComponent(0.5)
        scoreLabel.textAlignment = .center
        scoreLabel.layer.cornerRadius = 10
        scoreLabel.layer.masksToBounds = true

        view.addSubview(scoreLabel)

        NSLayoutConstraint.activate([
            scoreLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 16),
            scoreLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            scoreLabel.widthAnchor.constraint(equalToConstant: 150),
            scoreLabel.heightAnchor.constraint(equalToConstant: 50)
        ])
    }

    private func setupModel() {
        guard let mlModel = try? BallDetector(configuration: MLModelConfiguration()).model else {
            fatalError("Could not load BallDetector model")
        }
        visionModel = try? VNCoreMLModel(for: mlModel)
    }

    // MARK: - ARSessionDelegate
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard !isProcessingFrame else { return }
        isProcessingFrame = true

        // Process the current frame
        DispatchQueue.global(qos: .userInitiated).async {
            self.processFrame(pixelBuffer: frame.capturedImage)
        }
    }

    private func processFrame(pixelBuffer: CVPixelBuffer) {
        let request = VNCoreMLRequest(model: visionModel) { [weak self] request, error in
            guard let self = self else { return }
            defer { self.isProcessingFrame = false }
            guard let results = request.results as? [VNRecognizedObjectObservation] else { return }
            self.handleDetections(results)
        }

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        do {
            try handler.perform([request])
        } catch {
            print("Failed to perform request: \(error)")
            DispatchQueue.main.async {
                self.isProcessingFrame = false
            }
        }
    }

    private func handleDetections(_ detections: [VNRecognizedObjectObservation]) {
        DispatchQueue.main.async {
            self.sceneView.overlaySKScene?.removeAllChildren() // Clear old bounding boxes

            for detection in detections.prefix(1) { // Limit to 1 detection for performance
                let boundingBox = detection.boundingBox
                self.drawBoundingBox(boundingBox)

                // Project the bounding box center to AR space and check for goal crossing
                let screenPoint = CGPoint(
                    x: (boundingBox.origin.x + boundingBox.width / 2) * self.view.bounds.width,
                    y: (1 - (boundingBox.origin.y + boundingBox.height / 2)) * self.view.bounds.height
                )
                self.checkGoalCrossing(screenPoint: screenPoint)
            }
        }
    }


    private func drawBoundingBox(_ rect: CGRect) {
        let convertedRect = convertBoundingBox(rect: rect)
        let skView = sceneView.overlaySKScene ?? SKScene(size: sceneView.frame.size)
        sceneView.overlaySKScene = skView

        let box = SKShapeNode(rect: convertedRect)
        box.strokeColor = .red
        box.lineWidth = 2.0
        skView.addChild(box)
    }

    private func convertBoundingBox(rect: CGRect) -> CGRect {
        let x = rect.origin.y * view.bounds.width
        let y = (1 - rect.origin.x - rect.height) * view.bounds.height
        let width = rect.height * view.bounds.width
        let height = rect.width * view.bounds.height
        return CGRect(x: x, y: y, width: width, height: height)
    }

    @objc private func handleTap(_ gesture: UITapGestureRecognizer) {
        let location = gesture.location(in: sceneView)
        let hitTestResults = sceneView.hitTest(location, types: [.existingPlaneUsingExtent, .estimatedHorizontalPlane])

        if let result = hitTestResults.first {
            let position = SCNVector3(
                result.worldTransform.columns.3.x,
                result.worldTransform.columns.3.y,
                result.worldTransform.columns.3.z
            )
            print("Goal line added at: \(position)") // Debug log
            addGoalLine(at: position)
        } else {
            print("No hit test result for goal line placement") // Debug log
        }
    }

    private func addGoalLine(at position: SCNVector3) {
        let goalLine = SCNNode(geometry: SCNCylinder(radius: 0.01, height: 1.0))
        goalLine.geometry?.firstMaterial?.diffuse.contents = UIColor.red
        goalLine.eulerAngles.x = -.pi / 2
        goalLine.position = position
        sceneView.scene.rootNode.addChildNode(goalLine)
        goalLines.append(goalLine)
    }
    
    private func checkGoalCrossing(screenPoint: CGPoint) {
        let hitTestResults = sceneView.hitTest(screenPoint, types: .featurePoint)
        
        if let result = hitTestResults.first {
            let ballPosition = SCNVector3(
                result.worldTransform.columns.3.x,
                result.worldTransform.columns.3.y,
                result.worldTransform.columns.3.z
            )

            for goalLine in goalLines {
                let distance = (goalLine.position - ballPosition).length()
                
                // Check if the ball is close enough to the goal line
                if distance < 0.1 {
                    score += 1
                    scoreLabel.text = "Score: \(score)"
                    print("Goal scored! Current score: \(score)")
                    return // Exit after registering a goal
                }
            }
        }
    }

}

// MARK: - SCNVector3 Extension
extension SCNVector3 {
    static func - (left: SCNVector3, right: SCNVector3) -> SCNVector3 {
        return SCNVector3(left.x - right.x, left.y - right.y, left.z - right.z)
    }

    func length() -> Float {
        return sqrt(x * x + y * y + z * z)
    }
}
