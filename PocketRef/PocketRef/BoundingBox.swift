import SwiftUI

struct BoundingBox: View {
    var rect: CGRect

    var body: some View {
        GeometryReader { geometry in
            let convertedRect = CGRect(
                x: rect.origin.x * geometry.size.width,
                y: rect.origin.y * geometry.size.height,
                width: rect.size.width * geometry.size.width,
                height: rect.size.height * geometry.size.height
            )

            Rectangle()
                .stroke(Color.red, lineWidth: 2)
                .frame(
                    width: convertedRect.width,
                    height: convertedRect.height
                )
                .position(
                    x: convertedRect.midX,
                    y: convertedRect.midY
                )
        }
    }
}
