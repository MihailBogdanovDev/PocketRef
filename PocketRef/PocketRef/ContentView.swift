import SwiftUI

struct ContentView: View {
    var body: some View {
        ZStack {
            CameraView()
                .edgesIgnoringSafeArea(.all)
        }
    }
}

