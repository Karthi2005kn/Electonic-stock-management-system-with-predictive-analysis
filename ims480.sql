-- phpMyAdmin SQL Dump
-- version 5.1.3
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Sep 08, 2022 at 02:19 PM
-- Server version: 10.4.24-MariaDB
-- PHP Version: 7.4.29

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `ims480`
--

-- --------------------------------------------------------

--
-- Table structure for table `product`
--

CREATE TABLE `product` (
  `id` int(100) NOT NULL,
  `name` varchar(100) NOT NULL,
  `des` varchar(100) NOT NULL,
  `unit` int(100) NOT NULL,
  `unitprice` int(100) NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `product`
--


INSERT INTO `product` (`id`, `name`, `des`, `unit`, `unitprice`, `created_at`) VALUES
(13, 'Redmi Note10 Pro', '4k display,8GB RAM, 256GB Storage', 200, 25000, '2024-04-05 04:53:35'),
(14, 'HP Pavilion Laptop', 'Full HD display, 8GB RAM, 512 SSD, Intel i5 12th-Gen', 400, 70000, '2024-04-05 04:54:59'),
(15, 'Boat Headphones', 'Bluetooth, Noise cancellation', 200, 2000, '2024-04-05 04:56:32'),
(16, 'HP pendrive', '128GB', 500, 400, '2024-04-05 05:32:46'),
(17, 'Mi Power Bank', '20000 mAh', 200, 3000, '2024-04-05 05:36:38'),
(18, 'Boat Smart Watch', 'Bluetooth calling, Black rubber strap, Heartbeat moniter', 350, 3500, '2024-04-05 05:40:51'),
(19, 'DELL bluetooth mouse', 'Bluetooth, Black color', 500, 800, '2024-04-05 05:43:32'),
(22, 'Lenova ideapad 7', 'Full HD display, 16GB, 512 SSD', 150, 120000, '2024-04-05 06:08:56'),
(23, 'SanDisc Pendrive', '64GB', 550, 300, '2024-04-05 06:10:31'),
(24, 'LG OLED Smart TV', '64 inches, 4k display', 180, 70000, '2024-04-05 06:13:22'),
(25, 'Samsung M31', '8GB RAM,128GB Storage', 150, 20000, '2024-04-08 08:09:35'),
(29, 'IQOO Z7s', '5G, 8GB RAM, 128GB ROM', 200, 20000, '2024-04-12 03:23:39'),
(30, 'Keyboard', 'Wired, Fast process', 40, 450, '2024-04-12 04:22:40');


-- --------------------------------------------------------

--
-- Table structure for table `purchase`
--

CREATE TABLE `purchase` (
  `id` int(11) NOT NULL,
  `name` varchar(100) NOT NULL,
  `des` varchar(100) NOT NULL,
  `unit` int(100) NOT NULL,
  `unitprice` int(100) NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `purchase`
--

INSERT INTO `purchase` (`id`, `name`, `des`, `unit`, `unitprice`, `created_at`) VALUES
(0, 'Samsung LED TV', '48 inches', 100, 50000, '2024-04-05 04:52:31'),
(0, 'Redmi Note10 Pro', '4k display ', 200, 25000, '2024-04-05 04:53:35'),
(0, 'HP Pavilion Laptop', 'Full HD display, 512 SSD', 50, 70000, '2024-04-05 04:54:59'),
(0, 'Boat Headphones', 'Mp3 ', 300, 2000, '2024-04-05 04:56:32'),
(0, 'HP pendrive', '128GB', 500, 400, '2024-04-05 05:32:46'),
(0, 'Mi Power Bank', '20000 mAh', 200, 3000, '2024-04-05 05:36:38'),
(0, 'Boat Smart Watch', 'Bluetooth calling, Black rubber strap, Heartbeat moniter', 450, 3500, '2024-04-05 05:40:51'),
(0, 'DELL bluetooth mouse', 'Bluetooth, Black color', 500, 800, '2024-04-05 05:43:32'),
(0, 'Lenova ideapad 7', 'Full HD display, 16GB, 512 SSD', 250, 120000, '2024-04-05 06:08:56'),
(0, 'SanDisc Pendrive', '64GB', 700, 150, '2024-04-05 06:10:31'),
(0, 'LG OLED Smart TV', '64 inches, 4k display', 200, 70000, '2024-04-05 06:13:22'),
(0, 'Samsung M31', '8GB RAM,128GB Storage', 200, 20000, '2024-04-08 08:09:35'),
(0, 'Zebronics HeadPhone', 'Mp3', 10, 400, '2024-04-08 14:24:40'),
(0, 'IQOO Z7s', '5G, 8GB RAM, 128GB ROM', 250, 20000, '2024-04-12 03:23:39'),
(0, 'Keyboard', 'Wired, Fast process', 50, 500, '2024-04-12 04:22:40');

-- --------------------------------------------------------

--
-- Table structure for table `sales`
--

CREATE TABLE `sales` (
  `id` int(100) NOT NULL,
  `name` varchar(100) NOT NULL,
  `sellunit` int(100) NOT NULL,
  `totalprice` int(100) NOT NULL,
  `productid` int(100) NOT NULL,
  `created_at` timestamp(6) NOT NULL DEFAULT current_timestamp(6)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `sales`
--

INSERT INTO `sales` (`id`, `name`, `sellunit`, `totalprice`, `productid`, `created_at`) VALUES
(9, 'Lenova ideapad 7', 50, 6000000, 22, '2024-04-05 08:07:12.403463'),
(10, 'HP Pavilion Laptop', 100, 7000000, 14, '2024-04-05 08:07:21.430607'),
(11, 'Boat Smart Watch', 100, 350000, 18, '2024-04-05 08:07:28.538661'),
(12, 'Lenova ideapad 7', 50, 6000000, 22, '2024-04-05 08:07:35.519075'),
(13, 'SanDisc Pendrive', 150, 45000, 23, '2024-04-05 08:07:41.313114'),
(14, 'LG OLED Smart TV', 20, 1400000, 24, '2024-04-05 08:07:49.688723'),
(15, 'Samsung M31', 50, 1000000, 25, '2024-04-08 08:10:02.895024'),
(16, 'Zebronics HeadPhone', 6, 2400, 26, '2024-04-08 14:25:14.056011'),
(17, 'IQOO Z7s', 50, 1000000, 29, '2024-04-12 03:25:51.624791'),
(18, 'Keyboard', 10, 4500, 30, '2024-04-12 04:24:09.794714');


--
-- Indexes for dumped tables
--

--
-- Indexes for table `product`
--
ALTER TABLE `product`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `sales`
--
ALTER TABLE `sales`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `product`
--
ALTER TABLE `product`
  MODIFY `id` int(100) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=6;

--
-- AUTO_INCREMENT for table `sales`
--
ALTER TABLE `sales`
  MODIFY `id` int(100) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=6;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
