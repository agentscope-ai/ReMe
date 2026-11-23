import Foundation
import Network
import UniformTypeIdentifiers

/// Serves the bundled Vite build from a stable origin for WebKit storage and recording.
final class StudioAssets {
    static let host = "127.0.0.1"
    static let port: UInt16 = 47813
    static let origin = "http://\(host):\(port)"

    private let root: URL
    private let listener: NWListener
    private var connections: [ObjectIdentifier: NWConnection] = [:]
    private let headerEnd = Data([13, 10, 13, 10])

    init(root: URL) throws {
        self.root = root
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(
            host: .ipv4(.loopback), port: NWEndpoint.Port(rawValue: Self.port)!
        )
        listener = try NWListener(using: parameters)
    }

    func start(onStateChange: @escaping (NWListener.State) -> Void) {
        listener.stateUpdateHandler = onStateChange
        listener.newConnectionHandler = { [weak self] connection in
            guard let self, self.connections.count < 32 else {
                connection.cancel()
                return
            }
            self.connections[ObjectIdentifier(connection)] = connection
            connection.start(queue: .main)
            self.readRequest(from: connection, received: Data())
            DispatchQueue.main.asyncAfter(deadline: .now() + 10) { [weak self, weak connection] in
                if let connection { self?.close(connection) }
            }
        }
        listener.start(queue: .main)
    }

    deinit {
        listener.cancel()
        for connection in connections.values { connection.cancel() }
    }

    private func readRequest(from connection: NWConnection, received: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 8192) { [weak self] data, _, complete, error in
            guard let self else { return }
            let request = received + (data ?? Data())
            if error != nil || request.count > 16384 {
                self.close(connection)
            } else if let end = request.range(of: self.headerEnd) {
                self.serve(request[..<end.lowerBound], on: connection)
            } else if complete {
                self.close(connection)
            } else {
                self.readRequest(from: connection, received: request)
            }
        }
    }

    private func serve(_ request: Data, on connection: NWConnection) {
        guard let header = String(data: request, encoding: .utf8),
              let requestLine = header.components(separatedBy: "\r\n").first else {
            return respond("400 Bad Request", on: connection)
        }
        let fields = requestLine.split(separator: " ")
        guard fields.count == 3, fields[2] == "HTTP/1.1" || fields[2] == "HTTP/1.0" else {
            return respond("400 Bad Request", on: connection)
        }
        guard fields[0] == "GET" else { return respond("405 Method Not Allowed", on: connection) }
        let path = fields[1].prefix { $0 != "?" }
        let file: URL
        if path == "/" || path == "/index.html" {
            file = root.appendingPathComponent("index.html")
        } else if path.hasPrefix("/assets/") {
            let name = String(path.dropFirst("/assets/".count))
            let fileCharacters = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._")
            guard !name.isEmpty, !name.hasPrefix("."), name.unicodeScalars.allSatisfy(fileCharacters.contains) else {
                return respond("404 Not Found", on: connection)
            }
            file = root.appendingPathComponent("assets", isDirectory: true).appendingPathComponent(name)
        } else {
            return respond("404 Not Found", on: connection)
        }
        guard let body = try? Data(contentsOf: file, options: .mappedIfSafe) else {
            return respond("404 Not Found", on: connection)
        }
        let type = switch file.pathExtension {
        case "html": "text/html; charset=utf-8"
        case "css": "text/css"
        case "js": "text/javascript"
        default: UTType(filenameExtension: file.pathExtension)?.preferredMIMEType ?? "application/octet-stream"
        }
        respond("200 OK", on: connection, type: type, body: body)
    }

    private func respond(
        _ status: String, on connection: NWConnection, type: String = "text/plain", body: Data = Data()
    ) {
        let headers = [
            "HTTP/1.1 \(status)",
            "Content-Type: \(type)",
            "Content-Length: \(body.count)",
            "Connection: close",
            "Cache-Control: no-cache",
            "Content-Security-Policy: frame-ancestors 'none'; object-src 'none'; base-uri 'none'",
            "X-Content-Type-Options: nosniff",
            "", "",
        ].joined(separator: "\r\n")
        connection.send(content: Data(headers.utf8) + body, completion: .contentProcessed { [weak self] _ in
            self?.close(connection)
        })
    }

    private func close(_ connection: NWConnection) {
        if connections.removeValue(forKey: ObjectIdentifier(connection)) != nil { connection.cancel() }
    }
}
