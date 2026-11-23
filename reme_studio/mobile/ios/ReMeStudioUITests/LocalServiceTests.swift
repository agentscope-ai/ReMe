import XCTest

final class LocalServiceTests: XCTestCase {
    func testConnectsOnLaunchAndAfterRestart() {
        let app = XCUIApplication()
        let endpoint = ProcessInfo.processInfo.environment["REME_TEST_ENDPOINT"] ?? "http://127.0.0.1:2333"
        continueAfterFailure = false
        app.launch()
        let web = openGeneralSettings(app)
        let address = labeled(web.textFields, "ReMe 服务地址", "ReMe service URL")
        XCTAssertTrue(address.waitForExistence(timeout: 10))
        if let current = address.value as? String, current != endpoint {
            address.coordinate(withNormalizedOffset: CGVector(dx: 0.98, dy: 0.5)).tap()
            address.typeText(String(repeating: XCUIKeyboardKey.delete.rawValue, count: current.count) + endpoint)
            XCTAssertEqual(address.value as? String, endpoint)
            labeled(web.buttons, "连接", "Connect").tap()
        }
        assertConnected(web, endpoint: endpoint)

        app.terminate()
        app.launch()
        assertConnected(openGeneralSettings(app), endpoint: endpoint)
    }

    private func openGeneralSettings(_ app: XCUIApplication) -> XCUIElement {
        let web = app.webViews.firstMatch
        let navigation = labeled(web.buttons, "页面导航", "Navigation")
        XCTAssertTrue(navigation.waitForExistence(timeout: 30), app.debugDescription)
        navigation.tap()
        let settings = labeled(web.buttons, "设置", "Settings")
        XCTAssertTrue(settings.waitForExistence(timeout: 10))
        settings.tap()
        let general = labeled(web.buttons, "通用", "General")
        XCTAssertTrue(general.waitForExistence(timeout: 10), app.debugDescription)
        general.tap()
        return web
    }

    private func assertConnected(_ web: XCUIElement, endpoint: String) {
        let connected = labeled(web.staticTexts, "已连接", "Connected")
        XCTAssertTrue(connected.waitForExistence(timeout: 20), web.debugDescription)
        XCTAssertEqual(labeled(web.textFields, "ReMe 服务地址", "ReMe service URL").value as? String, endpoint)
    }

    private func labeled(_ query: XCUIElementQuery, _ chinese: String, _ english: String) -> XCUIElement {
        query.matching(NSPredicate(format: "label IN %@", [chinese, english])).firstMatch
    }
}
