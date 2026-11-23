import SwiftUI

@main
struct ReMeStudioApp: App {
    var body: some Scene {
        WindowGroup {
            StudioView().ignoresSafeArea(.container)
        }
    }
}

private struct StudioView: UIViewControllerRepresentable {
    func makeUIViewController(context: Context) -> StudioViewController {
        StudioViewController()
    }

    func updateUIViewController(_ controller: StudioViewController, context: Context) {}
}
