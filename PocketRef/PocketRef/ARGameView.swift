import SwiftUI
import ARKit
import SceneKit

protocol CameraViewControllerDelegate: AnyObject {
    func didDetectBall(at screenPosition: CGPoint)
}

struct ARGameView: UIViewControllerRepresentable {
    @Binding var score: Int

    // MARK: - Create the UIViewController
    func makeUIViewController(context: Context) -> ARGameViewController {
        ARGameViewController(score: $score)
    }

    // MARK: - Update the UIViewController
    func updateUIViewController(_ uiViewController: ARGameViewController, context: Context) {
        // Update ARGameViewController if needed (e.g., pass new data)
    }
}

class ARGameViewController: UIViewController, ARSCNViewDelegate, ARSessionDelegate, CameraViewControllerDelegate {
    private var sceneView: ARSCNView!
    private var goalLines: [SCNNode] = []
    @Binding var score: Int
    private var scoreLabel: UILabel!

    private var visionModel: VNCoreMLModel!

    init(score: Binding<Int>) {
        self._score = score
        super.init(nibName: nil, bundle: nil)
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func viewDidLoad() {
        super.viewDidLoad()

        // Load Core ML Model
        setupModel()

        // Setup AR Scene View
        sceneView = ARSCNView(frame: view.bounds)
        sceneView.delegate = self
        sceneView.scene = SCNScene()
        sceneView.session.delegate = self // Set ARSessionDelegate
        view.addSubview(sceneView)

        // Configure AR Session
        let configuration = ARWorldTrackingConfiguration()
        configuration.planeDetection = [.horizontal, .vertical]
        //configuration.frameSemantics = .sceneDepth // Optional for advanced features
        sceneView.session.run(configuration)

        // Gesture for goal line placement
        let tapGesture = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
        sceneView.addGestureRecognizer(tapGesture)

        // Add score label
        setupScoreLabel()
    }

    private func setupModel() {
        guard let mlModel = try? BallDetector(configuration: MLModelConfiguration()).model else {
            fatalError("Could not load BallDetector model")
        }
        visionModel = try? VNCoreMLModel(for: mlModel)
    }

    // MARK: - ARSessionDelegate
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let pixelBuffer = frame.capturedImage
        processFrame(pixelBuffer: pixelBuffer)
    }

    private func processFrame(pixelBuffer: CVPixelBuffer) {
        let request = VNCoreMLRequest(model: visionModel) { [weak self] request, error in
            guard let results = request.results as? [VNRecognizedObjectObservation] else {
                return
            }
            self?.handleDetections(results)
        }

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        do {
            try handler.perform([request])
        } catch {
            print("Failed to process frame: \(error)")
        }
    }

    private func handleDetections(_ detections: [VNRecognizedObjectObservation]) {
        DispatchQueue.main.async {
            for detection in detections {
                let boundingBox = detection.boundingBox
                let screenPoint = CGPoint(
                    x: (boundingBox.origin.x + boundingBox.width / 2) * self.view.bounds.width,
                    y: (1 - (boundingBox.origin.y + boundingBox.height / 2)) * self.view.bounds.height
                )

                self.didDetectBall(at: screenPoint)
            }
        }
    }

    // MARK: - CameraViewControllerDelegate
    func didDetectBall(at screenPosition: CGPoint) {
        let hitTestResults = sceneView.hitTest(screenPosition, types: .featurePoint)

        if let result = hitTestResults.first {
            let ballPosition = SCNVector3(
                result.worldTransform.columns.3.x,
                result.worldTransform.columns.3.y,
                result.worldTransform.columns.3.z
            )

            if let ballNode = sceneView.scene.rootNode.childNode(withName: "ballNode", recursively: false) {
                ballNode.position = ballPosition
            } else {
                let ballNode = SCNNode(geometry: SCNSphere(radius: 0.05))
                ballNode.geometry?.firstMaterial?.diffuse.contents = UIColor.green
                ballNode.name = "ballNode"
                ballNode.position = ballPosition
                sceneView.scene.rootNode.addChildNode(ballNode)
            }

            checkBallCrossing(ballPosition: ballPosition)
        }
    }

    // MARK: - Ball Crossing Check
    func checkBallCrossing(ballPosition: SCNVector3) {
        for goalLine in goalLines {
            let distance = (goalLine.position - ballPosition).length()
            if distance < 0.1 {
                score += 1
                updateScore()
            }
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

    @objc func handleTap(_ gesture: UITapGestureRecognizer) {
        let location = gesture.location(in: sceneView)
        let hitTestResults = sceneView.hitTest(location, types: [.existingPlaneUsingExtent])

        if let result = hitTestResults.first {
            let position = SCNVector3(
                result.worldTransform.columns.3.x,
                result.worldTransform.columns.3.y,
                result.worldTransform.columns.3.z
            )
            addGoalLine(at: position)
        }
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

    private func updateScore() {
        scoreLabel.text = "Score: \(score)"
    }
}


/*extension SCNVector3 {
    // Subtraction operator for SCNVector3
    static func - (left: SCNVector3, right: SCNVector3) -> SCNVector3 {
        return SCNVector3(
            left.x - right.x,
            left.y - right.y,
            left.z - right.z
        )
    }

    // Length (magnitude) of the vector
    func length() -> Float {
        return sqrt(x * x + y * y + z * z)
    }
}*/
