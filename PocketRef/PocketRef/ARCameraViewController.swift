import UIKit
import ARKit
import Vision

class ARCameraViewController: UIViewController, ARSCNViewDelegate, ARSessionDelegate {
    private var sceneView: ARSCNView!
    private var goalLines: [(start: SCNVector3, end: SCNVector3)] = []
    
    //Scoring
    private var scoreLabel: UILabel!
    private var score: Int = 0
    private var teamScores: [Int] = [0, 0] // Two teams max
    private var isGameActive: Bool = false // Track if the game is active
    private var currentTeams: Int = 0 // Number of active teams (1 or 2)


    // Core ML Model
    private var visionModel: VNCoreMLModel!
    private var isProcessingFrame = false
    
    //Goal Line setup
    private var firstPoint: SCNVector3?
    private var previewLineNode: SCNNode?
    private var previewSphereNode: SCNNode?
    private var focusCircleNode: SCNNode?
    private var secondPointPreviewNode: SCNNode?
    

    override func viewDidLoad() {
        super.viewDidLoad()

        // Setup AR Scene
        setupARScene()

        //Setup Ready Button
        setupReadyButton()

        // Setup Core ML Model
        setupModel()

        // Add gesture recognizer for placing goal points
        let tapGesture = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
        sceneView.addGestureRecognizer(tapGesture)
    }

    private func setupARScene() {
        sceneView = ARSCNView(frame: view.bounds)
        sceneView.delegate = self
        sceneView.session.delegate = self
        sceneView.scene = SCNScene()
        view.addSubview(sceneView)

        let configuration = ARWorldTrackingConfiguration()
        configuration.planeDetection = [.horizontal, .vertical] // Detect horizontal and vertical planes
        configuration.isLightEstimationEnabled = true // Enable light estimation
        sceneView.session.run(configuration)

        addFocusCircle() // Add focus indicator like in Measure app
    }
    
    private func setupReadyButton() {
        let readyButton = UIButton(type: .system)
        readyButton.translatesAutoresizingMaskIntoConstraints = false
        readyButton.setTitle("Ready", for: .normal)
        readyButton.backgroundColor = UIColor.systemBlue
        readyButton.tintColor = .white
        readyButton.layer.cornerRadius = 10
        readyButton.addTarget(self, action: #selector(handleReadyButton), for: .touchUpInside)
        
        view.addSubview(readyButton)
        
        NSLayoutConstraint.activate([
            readyButton.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -16),
            readyButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            readyButton.widthAnchor.constraint(equalToConstant: 100),
            readyButton.heightAnchor.constraint(equalToConstant: 50)
        ])
    }

    @objc private func handleReadyButton() {
        guard !goalLines.isEmpty else {
            print("At least one goal line is required to start the game.")
            return
        }
        
        isGameActive = true
        currentTeams = min(goalLines.count, 2)
        setupScoreLabels()
        print("Game started with \(currentTeams) team(s).")
    }
    private func setupScoreLabels() {
        for i in 0..<currentTeams {
            let scoreLabel = UILabel()
            scoreLabel.translatesAutoresizingMaskIntoConstraints = false
            scoreLabel.text = "Team \(i + 1): \(teamScores[i])"
            scoreLabel.font = UIFont.boldSystemFont(ofSize: 18)
            scoreLabel.textColor = .white
            scoreLabel.backgroundColor = UIColor.black.withAlphaComponent(0.5)
            scoreLabel.textAlignment = .center
            scoreLabel.layer.cornerRadius = 10
            scoreLabel.layer.masksToBounds = true
            
            view.addSubview(scoreLabel)
            
            NSLayoutConstraint.activate([
                scoreLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: CGFloat(16 + i * 60)),
                scoreLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 16),
                scoreLabel.widthAnchor.constraint(equalToConstant: 150),
                scoreLabel.heightAnchor.constraint(equalToConstant: 40)
            ])
        }
    }


    private func setupModel() {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all // Use all available units (CPU, GPU, Neural Engine)

        guard let mlModel = try? BallDetector(configuration: configuration).model else {
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

        // Update the preview line if firstPoint is set
        if let firstPoint = firstPoint {
            updatePreviewLine(to: frame)
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
        let hitTestResults = sceneView.hitTest(location, types: [.existingPlaneUsingExtent])

        if let result = hitTestResults.first {
            let position = SCNVector3(
                result.worldTransform.columns.3.x,
                result.worldTransform.columns.3.y,
                result.worldTransform.columns.3.z
            )

            if firstPoint == nil {
                // Set the first point
                firstPoint = position
                print("First point set at: \(position)")

                // Add a sphere node to indicate the first point
                addPreviewSphere(at: position)
            } else {
                // Set the second point, finalize the goal line, and reset
                let secondPoint = position
                print("Second point set at: \(secondPoint)")
                addGoalLine(start: firstPoint!, end: secondPoint)
                firstPoint = nil
                removePreviewLine()
                removePreviewSphere()
            }
        } else {
            print("No hit test result for goal line placement")
        }
    }


    private func addGoalLine(start: SCNVector3, end: SCNVector3) {
        // Create a line geometry between two points
        let lineGeometry = createLineGeometry(start: start, end: end)
        let lineNode = SCNNode(geometry: lineGeometry)
        lineNode.geometry?.firstMaterial?.diffuse.contents = UIColor.green
        sceneView.scene.rootNode.addChildNode(lineNode)

        goalLines.append((start: start, end: end))
    }
    
    private func createLineGeometry(start: SCNVector3, end: SCNVector3) -> SCNGeometry {
        let vertices: [SCNVector3] = [start, end]
        let vertexSource = SCNGeometrySource(vertices: vertices)
        
        let indices: [UInt16] = [0, 1] // Connect the two points
        let indexData = Data(bytes: indices, count: MemoryLayout<UInt16>.size * indices.count)
        let element = SCNGeometryElement(data: indexData,
                                         primitiveType: .line,
                                         primitiveCount: 1,
                                         bytesPerIndex: MemoryLayout<UInt16>.size)

        return SCNGeometry(sources: [vertexSource], elements: [element])
    }

    private func addPreviewSphere(at position: SCNVector3) {
        let sphereGeometry = SCNSphere(radius: 0.01)
        sphereGeometry.firstMaterial?.diffuse.contents = UIColor.red
        let sphereNode = SCNNode(geometry: sphereGeometry)
        sphereNode.position = position
        sceneView.scene.rootNode.addChildNode(sphereNode)
        previewSphereNode = sphereNode
    }

    private func addPreviewLine(start: SCNVector3) {
        let previewLine = SCNCylinder(radius: 0.002, height: 0.001)
        previewLine.firstMaterial?.diffuse.contents = UIColor.yellow.withAlphaComponent(0.8)
        let lineNode = SCNNode(geometry: previewLine)
        lineNode.position = start
        sceneView.scene.rootNode.addChildNode(lineNode)
        previewLineNode = lineNode
    }

    private func updatePreviewLine(to frame: ARFrame) {
        guard let firstPoint = firstPoint else { return }

        let hitTestResults = sceneView.hitTest(view.center, types: [.existingPlaneUsingExtent])
        if let result = hitTestResults.first {
            let currentPoint = SCNVector3(
                result.worldTransform.columns.3.x,
                result.worldTransform.columns.3.y,
                result.worldTransform.columns.3.z
            )

            // Create or update the preview line
            if previewLineNode == nil {
                previewLineNode = SCNNode(geometry: createLineGeometry(start: firstPoint, end: currentPoint))
                previewLineNode?.geometry?.firstMaterial?.diffuse.contents = UIColor.yellow
                sceneView.scene.rootNode.addChildNode(previewLineNode!)
            } else if let lineGeometry = previewLineNode?.geometry as? SCNGeometry {
                // Update the preview line's geometry with the new endpoint
                previewLineNode?.geometry = createLineGeometry(start: firstPoint, end: currentPoint)
            }
        }
    }

    private func addFocusCircle() {
        let focusGeometry = SCNTorus(ringRadius: 0.01, pipeRadius: 0.001)
        focusGeometry.firstMaterial?.diffuse.contents = UIColor.white.withAlphaComponent(0.8)
        let focusNode = SCNNode(geometry: focusGeometry)
        focusNode.position = SCNVector3(0, 0, -0.2)
        focusNode.eulerAngles.x = -.pi / 2
        sceneView.pointOfView?.addChildNode(focusNode)
        focusCircleNode = focusNode
    }

    private func removePreviewLine() {
        previewLineNode?.removeFromParentNode()
        previewLineNode = nil
    }

    private func removePreviewSphere() {
        previewSphereNode?.removeFromParentNode()
        previewSphereNode = nil
    }

    func renderer(_ renderer: SCNSceneRenderer, updateAtTime time: TimeInterval) {
        updateFocusCircle()
    }

    private func updateFocusCircle() {
        guard let focusCircleNode = focusCircleNode else { return }

        let hitTestResults = sceneView.hitTest(view.center, types: [.existingPlaneUsingExtent])
        if let result = hitTestResults.first {
            let position = SCNVector3(
                result.worldTransform.columns.3.x,
                result.worldTransform.columns.3.y,
                result.worldTransform.columns.3.z
            )
            focusCircleNode.position = position
        }
    }

    private func checkGoalCrossing(screenPoint: CGPoint) {
        let hitTestResults = sceneView.hitTest(screenPoint, types: .featurePoint)

        if let result = hitTestResults.first {
            let ballPosition = SCNVector3(
                result.worldTransform.columns.3.x,
                result.worldTransform.columns.3.y,
                result.worldTransform.columns.3.z
            )
            
            for (index, line) in goalLines.enumerated() {
                let (start, end) = line
                let distanceToLine = distanceFromPointToLine(point: ballPosition, lineStart: start, lineEnd: end)
                
                // Debugging logs
                print("Ball position: \(ballPosition), Start: \(start), End: \(end), Distance: \(distanceToLine)")
                
                if distanceToLine < 0.1 && ballBeyondLine(ball: ballPosition, lineStart: start, lineEnd: end) {
                    // Award a point to the opposing team
                    let scoringTeam = (index == 0) ? 1 : 0
                    teamScores[scoringTeam] += 1
                    print("Goal scored for Team \(scoringTeam + 1)! Scores: \(teamScores)")
                    
                    // Update score label
                    updateScoreLabel(for: scoringTeam)
                    
                    // Add feedback
                    addGoalFeedback(at: ballPosition)
                    
                    return // Exit after registering a goal
                }
            }
        }
    }
    
    private func updateScoreLabel(for team: Int) {
        if let scoreLabel = view.subviews.compactMap({ $0 as? UILabel }).first(where: { $0.text?.contains("Team \(team + 1)") == true }) {
            scoreLabel.text = "Team \(team + 1): \(teamScores[team])"
        }
    }

    
    private func addGoalFeedback(at position: SCNVector3) {
        let sphere = SCNSphere(radius: 0.05)
        sphere.firstMaterial?.diffuse.contents = UIColor.green
        let node = SCNNode(geometry: sphere)
        node.position = position
        sceneView.scene.rootNode.addChildNode(node)
        
        // Remove the node after 1 second
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            node.removeFromParentNode()
        }
    }



    private func ballBeyondLine(ball: SCNVector3, lineStart: SCNVector3, lineEnd: SCNVector3) -> Bool {
        let lineVector = lineEnd - lineStart // Vector representing the line
        let ballVector = ball - lineStart   // Vector from the start of the line to the ball
        
        let projectionFactor = ballVector.dot(lineVector) / lineVector.lengthSquared()
        
        // Check if the ball is beyond the line segment
        return projectionFactor > 1.0 || projectionFactor < 0.0
    }


    private func distanceFromPointToLine(point: SCNVector3, lineStart: SCNVector3, lineEnd: SCNVector3) -> Float {
        let lineVector = lineEnd - lineStart
        let pointVector = point - lineStart
        let lineLengthSquared = lineVector.lengthSquared()
        
        let projectionFactor = max(0, min(1, pointVector.dot(lineVector) / lineLengthSquared))
        let closestPointOnLine = lineStart + (lineVector * projectionFactor)
        return (point - closestPointOnLine).length()
    }
}

// MARK: - SCNVector3 Extension
extension SCNVector3 {
    static func - (left: SCNVector3, right: SCNVector3) -> SCNVector3 {
        return SCNVector3(left.x - right.x, left.y - right.y, left.z - right.z)
    }

    static func * (vector: SCNVector3, scalar: Float) -> SCNVector3 {
        return SCNVector3(vector.x * scalar, vector.y * scalar, vector.z * scalar)
    }

    static func + (left: SCNVector3, right: SCNVector3) -> SCNVector3 {
        return SCNVector3(left.x + right.x, left.y + right.y, left.z + right.z)
    }

    func dot(_ other: SCNVector3) -> Float {
        return x * other.x + y * other.y + z * other.z
    }

    func length() -> Float {
        return sqrt(x * x + y * y + z * z)
    }
    
    func lengthSquared() -> Float {
        return x * x + y * y + z * z
    }
}
