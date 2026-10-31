// SPDX-License-Identifier: GPL-3.0-or-later
// Links actual ORB-SLAM3; no replacement pose-estimation implementation.
#include "System.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <thread>

cv::Mat read_pgm(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    std::string magic;
    int width, height, maximum;
    file >> magic >> width >> height >> maximum;
    if (!file || magic != "P5" || maximum != 255 || width <= 0 || height <= 0)
        throw std::runtime_error("Expected an 8-bit binary PGM: " + path);
    file.get();
    cv::Mat image(height, width, CV_8U);
    file.read(reinterpret_cast<char*>(image.data), width*height);
    if (!file) throw std::runtime_error("Truncated image: " + path);
    return image;
}

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "usage: orbslam3_runner vocabulary settings mono|stereo frames.tsv poses.csv\n";
        return 2;
    }
    try {
        cv::setNumThreads(1);
        const std::string mode = argv[3];
        if (mode != "mono" && mode != "stereo") throw std::runtime_error("Unsupported sensor mode");
        ORB_SLAM3::System system(argv[1], argv[2], mode == "mono" ? ORB_SLAM3::System::MONOCULAR : ORB_SLAM3::System::STEREO, false);
        const std::string manifest = argv[4];
        const std::string directory = manifest.substr(0, manifest.find_last_of('/')) + "/";
        std::ifstream input(manifest);
        std::ofstream output(argv[5]);
        std::ofstream points(std::string(argv[5])+".points.csv");
        if (!input || !output || !points) throw std::runtime_error("Cannot open input/output manifest");
        points << "timestamp,map_id,point_id,x,y,z\n" << std::setprecision(17);
        output << "timestamp,tracking_state,map_id,tracked_map_points,latency_ms";
        for (int i = 0; i < 16; ++i) output << ",T" << i;
        output << "\n" << std::setprecision(17);
        std::string line;
        while (std::getline(input, line)) {
            if (line.empty()) continue;
            std::istringstream row(line);
            std::string stamp, left, right;
            std::getline(row, stamp, '\t');
            std::getline(row, left, '\t');
            std::getline(row, right, '\t');
            const double timestamp = std::stod(stamp);
            cv::Mat image = read_pgm(directory+left);
            const auto started = std::chrono::steady_clock::now();
            const Sophus::SE3f camera = mode == "mono" ? system.TrackMonocular(image, timestamp)
                : system.TrackStereo(image, read_pgm(directory+right), timestamp);
            const double elapsed_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
            const auto world = camera.inverse().matrix();
            int tracked = 0;
            long map_id = -1;
            for (auto point : system.GetTrackedMapPoints()) if (point && !point->isBad()) {
                ++tracked;
                map_id = point->GetMap()->GetId();
                const auto position = point->GetWorldPos();
                points << timestamp << ',' << point->GetMap()->GetId() << ',' << point->mnId << ','
                       << position[0] << ',' << position[1] << ',' << position[2] << '\n';
            }
            output << timestamp << ',' << system.GetTrackingState() << ',' << map_id << ',' << tracked << ',' << elapsed_ms;
            for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) output << ',' << world(r,c);
            output << '\n';
            // Feed a short sequence at camera pace while original mapping/loop workers run.
            std::this_thread::sleep_for(std::chrono::milliseconds(35));
        }
        system.Shutdown();
        system.SaveKeyFrameTrajectoryTUM(std::string(argv[5])+".keyframes.tum");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
