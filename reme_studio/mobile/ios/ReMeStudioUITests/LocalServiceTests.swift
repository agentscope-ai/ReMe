import XCTest

final class LocalServiceTests: XCTestCase {
    func testConnectsOnLaunchAndAfterRestart() {
        let app = XCUIApplication()
        let endpoint = ProcessInfo.processInfo.environment["REME_TEST_ENDPOINT"] ?? "http://127.0.0.1:2333"
        continueAfterFailure = false
        app.launch()
        assertConnected(openConnection(app, endpoint: endpoint), endpoint: endpoint)
        app.terminate()
        app.launch()
        assertConnected(openConnection(app), endpoint: endpoint)
    }

    func testBottomNavigationKeepsRemyCentralAndUpdatesInSettings() {
        let app = XCUIApplication()
        continueAfterFailure = false
        app.launch()
        let endpoint = ProcessInfo.processInfo.environment["REME_TEST_ENDPOINT"] ?? "http://127.0.0.1:2333"
        let web = openConnection(app, endpoint: endpoint)
        let names = [("主动智能", "Proactive AI"), ("随记", "Notebook"), ("Remy", "Remy"),
                     ("记忆", "Memory"), ("设置", "Settings")]
        let tabs = names.map { labeled(web.buttons, $0.0, $0.1) }
        for tab in tabs { XCTAssertTrue(tab.exists, web.debugDescription) }
        let positions = tabs.map { $0.frame.midX }
        XCTAssertEqual(positions, positions.sorted())
        XCTAssertEqual(positions[2], web.frame.midX, accuracy: 8)

        tabs[0].tap()
        let allTasks = labeled(web.buttons, "全部任务", "All tasks")
        XCTAssertTrue(allTasks.waitForExistence(timeout: 10), web.debugDescription)
        XCTAssertTrue(labeled(web.staticTexts, "数据源", "Sources").waitForExistence(timeout: 20), web.debugDescription)
        capture(app, "active-overview")
        allTasks.tap()
        capture(app, "all-tasks")
        tabs[1].tap()
        XCTAssertTrue(labeled(web.switches, "待办", "Todos").waitForExistence(timeout: 10), web.debugDescription)
        tabs[3].tap()
        XCTAssertTrue(labeled(web.buttons, "工作区", "Workspace").waitForExistence(timeout: 10), web.debugDescription)
        tabs[4].tap()
        labeled(web.buttons, "后退", "Back").tap()
        let updates = labeled(web.buttons, "版本与更新", "Version & updates")
        XCTAssertTrue(updates.waitForExistence(timeout: 10), web.debugDescription)
        capture(app, "settings-home")
        updates.tap()
        XCTAssertTrue(labeled(web.buttons, "查看发行版本", "View releases").waitForExistence(timeout: 10))
        tabs[2].tap()
        XCTAssertTrue(web.textViews.firstMatch.waitForExistence(timeout: 10), web.debugDescription)
        capture(app, "remy-chat")
    }

    func testKeyboardKeepsTheComposerAboveItAndHidesPrimaryNavigation() throws {
        let app = XCUIApplication()
        continueAfterFailure = false
        app.launch()
        let endpoint = ProcessInfo.processInfo.environment["REME_TEST_ENDPOINT"] ?? "http://127.0.0.1:2333"
        let web = openConnection(app, endpoint: endpoint)
        labeled(web.buttons, "Remy", "Remy").tap()
        let input = web.textViews.firstMatch
        XCTAssertTrue(input.waitForExistence(timeout: 10), web.debugDescription)
        input.tap()
        guard app.keyboards.firstMatch.waitForExistence(timeout: 5) else {
            throw XCTSkip("The simulator has no onscreen keyboard")
        }
        XCTAssertFalse(labeled(web.buttons, "设置", "Settings").exists)
        XCTAssertLessThanOrEqual(input.frame.maxY, app.keyboards.firstMatch.frame.minY)
        capture(app, "remy-keyboard")
    }

    func testProactiveOverviewUsesVerticalStagesAndSharedDetails() {
        let app = XCUIApplication()
        continueAfterFailure = false
        app.launch()
        let endpoint = ProcessInfo.processInfo.environment["REME_TEST_ENDPOINT"] ?? "http://127.0.0.1:2333"
        let web = openConnection(app, endpoint: endpoint)
        labeled(web.buttons, "主动智能", "Proactive AI").tap()
        let names = [("数据源", "Sources"), ("事件收件箱", "Event inbox"), ("任务", "Tasks"),
                     ("Remy 收件箱", "Remy inbox"), ("提醒", "Reminders")]
        let stages = names.map { labeled(web.staticTexts, $0.0, $0.1) }
        for stage in stages {
            XCTAssertTrue(stage.waitForExistence(timeout: 20), web.debugDescription)
            XCTAssertGreaterThanOrEqual(stage.frame.minX, web.frame.minX)
            XCTAssertLessThanOrEqual(stage.frame.maxX, web.frame.maxX)
        }
        let positions = stages.map { $0.frame.midY }
        XCTAssertEqual(positions, positions.sorted())
        capture(app, "proactive-vertical-stages")

        labeled(web.buttons, "展开内容", "Expand content").tap()
        let collapse = labeled(web.buttons, "收起内容", "Collapse content")
        XCTAssertTrue(collapse.waitForExistence(timeout: 10), web.debugDescription)
        XCTAssertGreaterThan(stages[1].frame.midY, positions[1] + 40)
        capture(app, "proactive-expanded-stages")
        collapse.tap()

        labeled(web.buttons, "添加来源", "Add source").tap()
        let close = labeled(web.buttons, "关闭详情", "Close details")
        XCTAssertTrue(close.waitForExistence(timeout: 10), web.debugDescription)
        XCTAssertGreaterThanOrEqual(close.frame.height, 44)
        capture(app, "proactive-source-details")
        close.tap()
        XCTAssertTrue(labeled(web.buttons, "全部任务", "All tasks").waitForExistence(timeout: 10))
    }

    func testMemoryKeepsOneStableSourceControl() {
        let app = XCUIApplication()
        continueAfterFailure = false
        app.launch()
        let endpoint = ProcessInfo.processInfo.environment["REME_TEST_ENDPOINT"] ?? "http://127.0.0.1:2333"
        let web = openConnection(app, endpoint: endpoint)
        labeled(web.buttons, "记忆", "Memory").tap()
        let names = [("工作区", "Workspace"), ("每日记忆", "Daily"), ("知识库", "Knowledge"), ("记忆标签", "Tags")]
        let first = labeled(web.buttons, names[0].0, names[0].1)
        XCTAssertTrue(first.waitForExistence(timeout: 15), web.debugDescription)
        let initialY = first.frame.midY
        capture(app, "memory-workspace-navigation")
        for index in [1, 2, 3, 0] {
            labeled(web.buttons, names[index].0, names[index].1).tap()
            XCTAssertTrue(first.waitForExistence(timeout: 10), web.debugDescription)
            var right: CGFloat = 0
            for name in names {
                let matches = web.buttons.matching(NSPredicate(format: "label IN %@", [name.0, name.1]))
                XCTAssertEqual(matches.count, 1)
                let button = matches.firstMatch
                XCTAssertEqual(button.frame.midY, initialY, accuracy: 1)
                XCTAssertGreaterThanOrEqual(button.frame.height, 44)
                XCTAssertGreaterThanOrEqual(button.frame.minX, right)
                XCTAssertLessThanOrEqual(button.frame.maxX, web.frame.maxX)
                right = button.frame.maxX
            }
            if index == 1 { capture(app, "memory-daily-navigation") }
        }
    }

    func testMobileMenusRemainDismissibleDuringTransitions() {
        let app = XCUIApplication()
        continueAfterFailure = false
        app.launch()
        let endpoint = ProcessInfo.processInfo.environment["REME_TEST_ENDPOINT"] ?? "http://127.0.0.1:2333"
        let web = openConnection(app, endpoint: endpoint)
        labeled(web.buttons, "Remy", "Remy").tap()
        let menu = labeled(web.descendants(matching: .any), "对话操作", "Conversation actions")
        XCTAssertTrue(menu.waitForExistence(timeout: 10), web.debugDescription)
        menu.tap()
        let clear = labeled(web.descendants(matching: .any), "清空上下文", "Clear context")
        XCTAssertTrue(clear.waitForExistence(timeout: 10), web.debugDescription)
        clear.tap()
        let cancel = labeled(web.buttons, "取消", "Cancel")
        XCTAssertTrue(cancel.waitForExistence(timeout: 10), web.debugDescription)
        cancel.tap()

        menu.tap()
        XCTAssertTrue(clear.waitForExistence(timeout: 10), web.debugDescription)
        web.textViews.firstMatch.tap()
        let dismissed = XCTNSPredicateExpectation(
            predicate: NSPredicate { _, _ in !clear.exists }, object: nil
        )
        XCTAssertEqual(XCTWaiter.wait(for: [dismissed], timeout: 5), .completed, web.debugDescription)
        capture(app, "mobile-menu-dismissed")
    }

    // Opt in only with a dedicated backend: this test intentionally restarts it.
    func testManagesBackendThroughNativeRequests() throws {
        guard let endpoint = ProcessInfo.processInfo.environment["REME_TEST_ENDPOINT"] else {
            throw XCTSkip("Set REME_TEST_ENDPOINT to a configured ReMe test instance")
        }
        let app = XCUIApplication()
        continueAfterFailure = false
        app.launch()
        let web = openConnection(app, endpoint: endpoint)
        labeled(web.buttons, "后退", "Back").tap()
        labeled(web.buttons, "工作区", "Workspace").tap()
        let workspace = labeled(web.textFields, "工作区路径", "Workspace path")
        XCTAssertTrue(workspace.waitForExistence(timeout: 15), web.debugDescription)
        let ready = labeled(web.staticTexts, "后端已就绪", "Backend ready")
        XCTAssertTrue(ready.waitForExistence(timeout: 10))
        labeled(web.buttons, "重启后端", "Restart backend").tap()
        XCTAssertTrue(labeled(web.staticTexts, "正在重启后端…", "Restarting backend…").waitForExistence(timeout: 10))
        XCTAssertTrue(ready.waitForExistence(timeout: 25), web.debugDescription)
    }

    private func openConnection(_ app: XCUIApplication, endpoint: String? = nil) -> XCUIElement {
        let web = app.webViews.firstMatch
        let settings = labeled(web.buttons, "设置", "Settings")
        let address = labeled(web.textFields, "ReMe 服务地址", "ReMe service URL")
        XCTAssertTrue(web.waitForExistence(timeout: 30), app.debugDescription)
        let loaded = XCTNSPredicateExpectation(
            predicate: NSPredicate { _, _ in settings.exists || address.exists }, object: nil
        )
        XCTAssertEqual(XCTWaiter.wait(for: [loaded], timeout: 30), .completed, web.debugDescription)
        if !settings.exists, let endpoint, address.exists, address.value as? String != endpoint {
            connect(web, address: address, endpoint: endpoint)
        }
        XCTAssertTrue(settings.waitForExistence(timeout: 30), app.debugDescription)
        settings.tap()
        let connection = labeled(web.buttons, "连接", "Connection")
        XCTAssertTrue(connection.waitForExistence(timeout: 10), app.debugDescription)
        connection.tap()
        XCTAssertTrue(address.waitForExistence(timeout: 10), app.debugDescription)
        if let endpoint, address.value as? String != endpoint {
            connect(web, address: address, endpoint: endpoint)
        }
        return web
    }

    private func connect(_ web: XCUIElement, address: XCUIElement, endpoint: String) {
        let current = address.value as? String ?? ""
        if current != endpoint {
            address.coordinate(withNormalizedOffset: CGVector(dx: 0.98, dy: 0.5)).tap()
            address.typeText(String(repeating: XCUIKeyboardKey.delete.rawValue, count: current.count) + endpoint)
        }
        XCTAssertEqual(address.value as? String, endpoint)
        let submit = labeled(web.buttons, "连接 ReMe 服务", "Connect to ReMe service")
        XCTAssertTrue(submit.waitForExistence(timeout: 10), web.debugDescription)
        submit.tap()
    }

    private func assertConnected(_ web: XCUIElement, endpoint: String) {
        XCTAssertTrue(labeled(web.staticTexts, "已连接", "Connected").waitForExistence(timeout: 20), web.debugDescription)
        XCTAssertEqual(labeled(web.textFields, "ReMe 服务地址", "ReMe service URL").value as? String, endpoint)
    }

    private func capture(_ app: XCUIApplication, _ name: String) {
        let attachment = XCTAttachment(screenshot: XCUIScreen.main.screenshot())
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    private func labeled(_ query: XCUIElementQuery, _ chinese: String, _ english: String) -> XCUIElement {
        query.matching(NSPredicate(format: "label IN %@", [chinese, english])).firstMatch
    }
}
