import XCTest

final class LocalServiceTests: XCTestCase {
    func testConnectsOnLaunchAndAfterRestart() {
        let app = XCUIApplication()
        continueAfterFailure = false
        app.launch()
        assertConnected(app)

        app.terminate()
        app.launch()
        assertConnected(app)
    }

    private func assertConnected(_ app: XCUIApplication) {
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
        let connected = labeled(web.staticTexts, "已连接", "Connected")
        XCTAssertTrue(connected.waitForExistence(timeout: 20), app.debugDescription)
        XCTAssertFalse(labeled(web.textFields, "ReMe 服务地址", "ReMe service URL").exists)
    }

    private func labeled(_ query: XCUIElementQuery, _ chinese: String, _ english: String) -> XCUIElement {
        query.matching(NSPredicate(format: "label IN %@", [chinese, english])).firstMatch
    }
}
