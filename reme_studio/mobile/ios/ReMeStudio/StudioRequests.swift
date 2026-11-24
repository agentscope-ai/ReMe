import Foundation

@MainActor
final class StudioRequests {
    private final class Request {
        var opening: Task<Void, Never>?
        var task: URLSessionDataTask?
        var bytes: URLSession.AsyncBytes.Iterator?
    }

    private let session: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.httpShouldSetCookies = false
        config.urlCredentialStorage = nil
        // The shared client controls timeouts. A chat may wait on a long tool call.
        config.timeoutIntervalForRequest = config.timeoutIntervalForResource
        return URLSession(configuration: config)
    }()
    private var active: [Int: Request] = [:]

    deinit { session.invalidateAndCancel() }

    func open(id: Int, url: URL, body: String, accept: String, reply: @escaping (Any?, String?) -> Void) {
        cancel(id: id)
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.httpBody = Data(body.utf8)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if !accept.isEmpty { request.setValue(accept, forHTTPHeaderField: "Accept") }

        let entry = Request()
        active[id] = entry
        entry.opening = Task {
            defer { entry.opening = nil }
            do {
                let (bytes, response) = try await session.bytes(for: request)
                entry.task = bytes.task
                try Task.checkCancellation()
                guard let http = response as? HTTPURLResponse else { throw URLError(.badServerResponse) }
                entry.bytes = bytes.makeAsyncIterator()
                reply(["status": http.statusCode, "contentType": http.value(forHTTPHeaderField: "Content-Type") ?? ""], nil)
            } catch {
                entry.task?.cancel()
                if active[id] === entry { active.removeValue(forKey: id) }
                reply(nil, error.localizedDescription)
            }
        }
    }

    func read(id: Int, reply: @escaping (Any?, String?) -> Void) {
        guard let entry = active[id], var bytes = entry.bytes else {
            reply(nil, "ReMe response is unavailable")
            return
        }
        entry.bytes = nil
        Task {
            do {
                var chunk = Data()
                // Flush SSE lines as they arrive and bound the size of each bridge message.
                while chunk.count < 16 * 1024, let byte = try await bytes.next() {
                    chunk.append(byte)
                    if byte == 10 { break }
                }
                guard active[id] === entry else { throw URLError(.cancelled) }
                if chunk.isEmpty {
                    active.removeValue(forKey: id)
                    reply(NSNull(), nil)
                } else {
                    entry.bytes = bytes
                    reply(chunk.base64EncodedString(), nil)
                }
            } catch {
                if active[id] === entry { cancel(id: id) }
                reply(nil, error.localizedDescription)
            }
        }
    }

    func cancel(id: Int) {
        guard let entry = active.removeValue(forKey: id) else { return }
        entry.opening?.cancel()
        entry.task?.cancel()
    }

    func cancelAll() {
        for id in active.keys { cancel(id: id) }
    }
}
