import SwiftUI

struct GameStartView: View {
    @State private var isGameStarted: Bool = false

    var body: some View {
        VStack {
            if isGameStarted {
                ARCameraGameView()
            } else {
                Button(action: {
                    self.isGameStarted.toggle()
                }) {
                    Text("Start Game")
                        .font(.largeTitle)
                        .foregroundColor(.white)
                        .padding()
                        .background(Color.blue)
                        .cornerRadius(10)
                }
            }
        }
    }
}

struct ARCameraGameView: UIViewControllerRepresentable {
    func makeUIViewController(context: Context) -> ARCameraViewController {
        return ARCameraViewController()
    }

    func updateUIViewController(_ uiViewController: ARCameraViewController, context: Context) {}
}
