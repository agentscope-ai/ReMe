import XCTest

final class LocalServiceTests: XCTestCase {
    func testManagesBackendThroughNativeRequests() throws {
        guard let endpoint = ProcessInfo.processInfo.environment["REME_TEST_ENDPOINT"] else {
            throw XCTSkip("Set REME_TEST_ENDPOINT to a configured ReMe test instance")
        }
        let app = XCUIApplication()
        continueAfterFailure = false
        app.launch()
        let web = openGeneralSettings(app, endpoint: endpoint)
        let sidebar = labeled(web.buttons, "展开侧栏", "Show sidebar")
        if sidebar.waitForExistence(timeout: 3) {
            sidebar.tap()
        } else {
            tapNavigation(web)
            labeled(web.buttons, "设置", "Settings").tap()
        }
        let backend = labeled(web.buttons, "后端", "Backend")
        XCTAssertTrue(backend.waitForExistence(timeout: 10), web.debugDescription)
        backend.tap()
        let workspace = web.textFields.matching(
            NSPredicate(format: "label BEGINSWITH %@ OR label BEGINSWITH %@", "后端工作区路径", "Backend workspace path")
        ).firstMatch
        XCTAssertTrue(workspace.waitForExistence(timeout: 15), web.debugDescription)
        let ready = labeled(web.staticTexts, "后端已就绪", "Backend ready")
        XCTAssertTrue(ready.waitForExistence(timeout: 10))
        labeled(web.buttons, "重启后端", "Restart backend").tap()
        XCTAssertTrue(labeled(web.staticTexts, "正在重启后端…", "Restarting backend…").waitForExistence(timeout: 10))
        XCTAssertTrue(ready.waitForExistence(timeout: 25), web.debugDescription)
    }

    func testConnectsOnLaunchAndAfterRestart() {
        let app = XCUIApplication()
        let endpoint = ProcessInfo.processInfo.environment["REME_TEST_ENDPOINT"] ?? "http://127.0.0.1:2333"
        continueAfterFailure = false
        app.launch()
        let web = openGeneralSettings(app, endpoint: endpoint)
        assertConnected(web, endpoint: endpoint)

        app.terminate()
        app.launch()
        assertConnected(openGeneralSettings(app), endpoint: endpoint)
    }

    private func openGeneralSettings(_ app: XCUIApplication, endpoint: String? = nil) -> XCUIElement {
        let web = app.webViews.firstMatch
        let navigation = labeled(web.buttons, "页面导航", "Navigation")
        let address = labeled(web.textFields, "ReMe 服务地址", "ReMe service URL")
        let connectionPage = labeled(web.staticTexts, "连接 ReMe", "Connect to ReMe")
        let ready = XCTNSPredicateExpectation(
            predicate: NSPredicate { _, _ in connectionPage.exists || navigation.exists }, object: nil
        )
        XCTAssertEqual(XCTWaiter.wait(for: [ready], timeout: 30), .completed, app.debugDescription)
        if let endpoint, connectionPage.exists {
            XCTAssertTrue(address.waitForExistence(timeout: 10), web.debugDescription)
            connect(web, address: address, endpoint: endpoint)
        }
        let navigationReady = XCTNSPredicateExpectation(
            predicate: NSPredicate { _, _ in !connectionPage.exists && navigation.exists }, object: nil
        )
        XCTAssertEqual(XCTWaiter.wait(for: [navigationReady], timeout: 20), .completed, app.debugDescription)
        tapNavigation(web)
        let settings = labeled(web.buttons, "设置", "Settings")
        XCTAssertTrue(settings.waitForExistence(timeout: 10))
        settings.tap()
        let general = labeled(web.buttons, "通用", "General")
        XCTAssertTrue(general.waitForExistence(timeout: 10), app.debugDescription)
        general.tap()
        XCTAssertTrue(address.waitForExistence(timeout: 10), app.debugDescription)
        if let endpoint {
            connect(web, address: address, endpoint: endpoint)
        }
        return web
    }

    private func tapNavigation(_ web: XCUIElement) {
        labeled(web.buttons, "页面导航", "Navigation")
            .coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
    }

    private func connect(_ web: XCUIElement, address: XCUIElement, endpoint: String) {
        let current = address.value as? String ?? ""
        guard current != endpoint else { return }
        address.coordinate(withNormalizedOffset: CGVector(dx: 0.98, dy: 0.5)).tap()
        address.typeText(String(repeating: XCUIKeyboardKey.delete.rawValue, count: current.count) + endpoint)
        XCTAssertEqual(address.value as? String, endpoint)
        labeled(web.buttons, "连接", "Connect").tap()
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
